import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import study
from network import DRIVER, HUB, LINKS, connectivity, topology
from study import configuration
from tc_setup import delay_rules, rate_rules, verify


class NetworkStudyTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(
            study,
            "workload_host",
            return_value={"available_memory_bytes": 64 * 2**30, "total_memory_bytes": 128 * 2**30},
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_helper_identity_is_rejected_when_the_shared_core_release_changes(self):
        with (
            TemporaryDirectory() as folder,
            patch.object(study, "ROOT", Path(folder)),
            patch.object(study, "build_identity", return_value="source"),
        ):
            path = Path(folder) / ".runtime/image.json"
            study.save(
                path,
                {
                    "tag": "tools",
                    "id": "sha256:tools",
                    "core": {"tag": "et-ws-server-core:local", "id": "sha256:original"},
                    "build_inputs": "source",
                },
            )
            with (
                patch.object(
                    study,
                    "command",
                    side_effect=[SimpleNamespace(stdout="sha256:tools"), SimpleNamespace(stdout="sha256:new")],
                ),
                self.assertRaisesRegex(RuntimeError, "Shared Core release changed"),
            ):
                study.image_identity()

    def test_preparation_builds_only_tools_from_the_shared_release(self):
        core = {"Id": "sha256:base", "Architecture": "arm64"}
        tools = {"Id": "sha256:tools", "Architecture": "arm64"}
        results = [json.dumps([core]), json.dumps([core]), json.dumps([tools]), "rustc"]
        with (
            TemporaryDirectory() as folder,
            patch.object(study, "ROOT", Path(folder)),
            patch.object(study, "command", side_effect=[SimpleNamespace(stdout=value) for value in results]),
            patch.object(study.subprocess, "run") as build,
            patch.object(study, "build_identity", return_value="source"),
        ):
            study.prepare()
            recorded = json.loads((Path(folder) / ".runtime/image.json").read_text())
            self.assertEqual(recorded["core"]["tag"], "et-ws-server-core:local")
            self.assertEqual(recorded["core"]["id"], "sha256:base")
            self.assertIn("CORE_IMAGE=et-ws-server-core:local", build.call_args.args[0])

    def test_zero_agent_smoke_is_rejected_without_starting_containers(self):
        args = SimpleNamespace(command="smoke", agents=0, output=None, case=None)
        with patch.object(study, "image_identity", return_value={}), patch.object(study, "execute") as execute:
            with self.assertRaisesRegex(ValueError, "between 1 and 9000"):
                study.run(args)
            execute.assert_not_called()

    def test_unverified_previous_cleanup_blocks_another_attempt(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            report = output / "baseline-r01/attempt-01/report.json"
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps({"cleanup_verified": False}))
            args = SimpleNamespace(command="run", agents=None, output=output, case=None)
            with (
                patch.object(study, "image_identity", return_value={"build_inputs": study.build_identity()}),
                patch.object(study.platform, "system", return_value="Linux"),
                patch.object(
                    study,
                    "workload_host",
                    return_value={"total_memory_bytes": 64 * 2**30, "available_memory_bytes": 32 * 2**30},
                ),
                patch.object(study, "execute", side_effect=RuntimeError("unexpected retry")) as execute,
                self.assertRaisesRegex(RuntimeError, "Unverified cleanup"),
            ):
                study.run(args)
            execute.assert_not_called()

    def failed_setup(self, *, logs_error=None, down_error=None, verification_error=None):
        calls = []

        def compose(_project, _file, *args, **_kwargs):
            calls.append(args[0])
            if args[0] == "up":
                raise RuntimeError("simulated setup failure")
            if args[0] == "logs" and logs_error:
                raise logs_error
            if args[0] == "down" and down_error:
                raise down_error
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        def command(args, **_kwargs):
            if args[:2] == ["docker", "info"]:
                return SimpleNamespace(returncode=0, stdout="daemon", stderr="")
            if verification_error:
                raise verification_error
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with (
            TemporaryDirectory() as directory,
            patch.object(study, "topology", return_value={}),
            patch.object(study, "compose", side_effect=compose),
            patch.object(
                study,
                "command",
                side_effect=command,
            ),
        ):
            output = Path(directory) / "attempt"
            report = study.execute({"id": "baseline"}, {}, {"id": "fake"}, output)
            self.assertEqual(report, json.loads((output / "report.json").read_text()))
        self.assertIn("down", calls)
        self.assertIn("simulated setup failure", report["harness_error"])
        return report

    def test_log_timeout_does_not_skip_teardown_or_the_final_report(self):
        report = self.failed_setup(logs_error=subprocess.TimeoutExpired("logs", 30))
        self.assertTrue(report["cleanup_verified"])
        self.assertIn("log_collection_error", report)

    def test_teardown_timeout_still_verifies_absence_and_retains_the_timeout(self):
        report = self.failed_setup(down_error=subprocess.TimeoutExpired("down", 120))
        self.assertTrue(report["cleanup_verified"])
        self.assertIn("TimeoutExpired", report["cleanup_diagnostics"][0])

    def test_cleanup_verification_failure_remains_unverified(self):
        report = self.failed_setup(verification_error=subprocess.TimeoutExpired("ps", 30))
        self.assertFalse(report["cleanup_verified"])
        self.assertTrue(report["cleanup_errors"])

    def test_default_conditions_and_loss_window(self):
        config = configuration()
        cases = {case["id"]: case for case in config["cases"]}
        self.assertEqual(len(cases), 4)
        self.assertEqual(config["agents"], 25000)
        self.assertEqual(config["hold_seconds"], 120)
        self.assertEqual(cases["bandwidth-10kbit-per-vessel"]["rate_kbit"], 10)
        self.assertEqual(cases["latency-500ms-rtt"]["delay_ms_each_direction"], 250)
        loss = cases["loss-30pct"]
        self.assertEqual(loss["loss_percent"], 30)
        self.assertTrue(loss["loss_after_registration"] and loss["reconnect"])
        self.assertEqual(loss["observation"]["hold_seconds"], 600)
        self.assertNotIn("recovery_policy", config)
        effective = {**config, "tolerant_timeouts": True, "recovery_policy": loss["policy"]}
        env = topology("image", Path("/tmp/example"), effective)["services"]["hub"]["environment"]
        self.assertEqual(env["WS_CONNECTION_TIMEOUT"], "120s")
        self.assertEqual(env["HTTP_CLIENT_REQUEST_TIMEOUT"], "60s")

    def test_rate_limits_are_per_source_port_without_delayed_leaves(self):
        commands = rate_rules([20000, 20001, 45000], "driver", 10)
        self.assertEqual(sum("rate 10kbit ceil 10kbit" in row for row in commands), 3)
        self.assertEqual(sum("pfifo limit 64" in row for row in commands), 3)
        self.assertFalse(any("netem" in row for row in commands))
        self.assertTrue(any("sport 45000" in row for row in commands))
        self.assertTrue(any("hashkey mask 0x00ff0000" in row for row in commands))

    def test_reverse_rate_limits_match_vessel_destination(self):
        commands = rate_rules([20000], "hub", 10)
        self.assertTrue(any("dport 20000" in row for row in commands))
        self.assertTrue(any("hashkey mask 0x000000ff" in row for row in commands))

    def test_one_shared_impairment_queue_and_unimpaired_control(self):
        commands = delay_rules(500, 30, 65536)
        self.assertEqual(sum("netem" in row for row in commands), 1)
        self.assertTrue(any("netem limit 65536 delay 500ms loss random 30%" in row for row in commands))
        self.assertTrue(any("sport 50001" in row and "flowid 1:1" in row for row in commands))
        self.assertTrue(any("dport 50001" in row and "flowid 1:1" in row for row in commands))

    def test_live_delay_and_loss_must_match(self):
        data = {
            "qdisc": [
                {"kind": "netem", "options": {"delay": {"delay": 0.5}, "loss-random": {"loss": 0.3}, "limit": 65536}}
            ]
        }
        verify(data, "delay", [], 0, 500, 30, 65536)
        with self.assertRaises(RuntimeError):
            verify(data, "delay", [], 0, 250, 30, 65536)

    def test_private_underlay_and_separate_inner_links(self):
        document = topology("image", Path("/tmp/example"), {"hub_memory": "4g", "driver_memory": "8g"})
        services = document["services"]
        self.assertTrue(document["networks"]["underlay"]["internal"])
        self.assertEqual(
            [(left, right) for left, right, *_ in LINKS], [("driver", "rate"), ("rate", "delay"), ("delay", "hub")]
        )
        self.assertNotEqual(DRIVER, HUB)
        self.assertEqual(services["hub"]["cpus"], 1)
        for name in ("rate", "delay"):
            self.assertEqual(services[name]["sysctls"]["net.ipv4.ip_forward"], "1")
            self.assertEqual(services[name]["cpus"], 2)
        for service in services.values():
            self.assertNotIn("ports", service)
            self.assertNotIn("network_mode", service)
        self.assertTrue(all(service["networks"] == ["underlay"] for service in services.values()))

    def test_connectivity_failure_is_a_setup_error_with_evidence(self):
        calls = []

        def blocked(*args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(returncode=1, stdout="", stderr="timed out")

        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "connectivity failed"):
                connectivity(blocked, Path(directory), "before-shaping")
            evidence = json.loads((Path(directory) / "connectivity-before-shaping.json").read_text())
            self.assertEqual(evidence["exit_code"], 1)
            self.assertIn("timeout=3", calls[0][0][-1])


if __name__ == "__main__":
    unittest.main()
