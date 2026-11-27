"""Timing and lifecycle regressions that must not manufacture successful recovery evidence."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import study
from topology import Deployment, available_source_ports, document, save
from workload import Fleet, qualified_fleet, registered, summarize, until

TARGET = {"host": "unix:///test/docker.sock", "id": "daemon"}


class TimingTests(unittest.TestCase):
    """Keep unhealthy-observation timing distinct from gateway DOWN and fault injection."""

    def samples(self):
        return [
            {"at": 9.8, "check_status": "L7OK", "status": "UP"},
            {"at": 10.2, "check_status": "L4CON", "status": "UP 1/2"},
            {"at": 10.7, "check_status": "L4CON", "status": "DOWN"},
            {"at": 11.0, "check_status": "L4CON", "status": "DOWN"},
        ]

    def test_recovery_starts_at_unhealthy_observation(self):
        timing = summarize(self.samples(), 10, 11.2)
        self.assertAlmostEqual(timing["recovery_seconds"], 1)
        self.assertAlmostEqual(timing["detection_seconds"], 0.5)
        self.assertAlmostEqual(timing["post_down_seconds"], 0.5)
        self.assertAlmostEqual(timing["fault_to_work_seconds"], 1.2)

    def test_missing_detection_is_not_a_zero_second_recovery(self):
        with self.assertRaisesRegex(RuntimeError, "Missing"):
            summarize(self.samples()[:1], 10, 11.2)

    def test_missing_down_is_not_success(self):
        with self.assertRaisesRegex(RuntimeError, "Missing"):
            summarize(self.samples()[:2], 10, 11.2)

    def test_observer_gap_invalidates_timing(self):
        samples = self.samples()
        samples[-1]["at"] = 15
        with self.assertRaisesRegex(RuntimeError, "gap"):
            summarize(samples, 10, 16)

    def test_observer_error_invalidates_an_otherwise_complete_timing_window(self):
        with self.assertRaisesRegex(RuntimeError, "observer failed"):
            summarize(self.samples(), 10, 11.2, ["poll failed"])

    def test_complete_recovery_evidence_must_arrive_before_the_deadline(self):
        for recovered in (11.2, 11.3):
            with self.subTest(recovered=recovered), self.assertRaisesRegex(RuntimeError, "recovery deadline"):
                summarize(self.samples(), 10, recovered, deadline=11.2)
        self.assertAlmostEqual(summarize(self.samples(), 10, 11.1, deadline=11.2)["recovery_seconds"], 0.9)

    def test_gap_crossing_fault_injection_is_not_hidden(self):
        samples = [
            {"at": 9.8, "check_status": "L7OK", "status": "UP"},
            {"at": 14.0, "check_status": "L4CON", "status": "DOWN"},
            {"at": 14.2, "check_status": "L4CON", "status": "DOWN"},
        ]
        with self.assertRaisesRegex(RuntimeError, "gap"):
            summarize(samples, 10, 14.3)


class EvidenceTests(unittest.TestCase):
    """Only intact evidence for the same study may be resumed."""

    def test_prepare_reuses_the_shared_core_image_without_building_it(self):
        image = {"Id": "sha256:core", "Architecture": "arm64", "Config": {"Labels": {}}}
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(study, "ROOT", Path(folder)),
            patch.object(study, "command", return_value=json.dumps([image])) as command,
            patch.object(study, "resolve_docker", return_value=TARGET),
        ):
            save(Path(folder) / "settings.json", {"gateway_image": "haproxy"})
            study.prepare()
            recorded = json.loads((Path(folder) / ".runtime/snapshot.json").read_text())
            self.assertEqual(recorded["image"], "et-ws-server-core:local")
            self.assertEqual(recorded["image_id"], "sha256:core")
            self.assertEqual(
                command.call_args_list[0].args[0],
                ["docker", "--host", TARGET["host"], "image", "inspect", "et-ws-server-core:local"],
            )
            self.assertEqual(command.call_args_list[1].args[0], ["docker", "--host", TARGET["host"], "pull", "haproxy"])

    def test_full_fleet_claim_requires_fresh_heartbeats_on_the_expected_hub(self):
        sample = {"connected": 25000, "recent_heartbeats": 25000, "on_a": 25000, "on_b": 0}
        self.assertTrue(qualified_fleet(sample, 25000, "a"))
        self.assertFalse(qualified_fleet(sample, 25000, "b"))
        self.assertFalse(qualified_fleet({**sample, "recent_heartbeats": 24999}, 25000, "a"))
        self.assertFalse(qualified_fleet({**sample, "on_a": 24999}, 25000, "a"))

    def test_checked_source_ports_exclude_reserved_ranges_and_respect_the_pool_boundary(self):
        import socket

        pool = available_source_ports(21000, 3, stop=32768)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", pool[0]))
            ports = available_source_ports(pool[0], 1, stop=pool[2] + 1, excluded={pool[1]})
            self.assertNotIn(ports[0], (pool[0], pool[1]))
            self.assertLessEqual(ports[0], pool[2])
        with self.assertRaisesRegex(RuntimeError, "required"):
            available_source_ports(21000, 1, stop=21002, excluded={21000, 21001})

    def test_relative_evidence_path_produces_absolute_compose_mounts(self):
        import os

        with tempfile.TemporaryDirectory(dir=".") as folder:
            relative = Path(os.path.relpath(folder))
            settings = json.loads((study.ROOT / "settings.json").read_text())
            project = Deployment(settings, "image", relative, docker_target=TARGET)
            self.assertTrue(project.directory.is_absolute())
            document = json.loads((project.directory / "compose.json").read_text())
            self.assertTrue(document["services"]["gateway"]["volumes"][0].startswith(str(relative.resolve())))

    def test_tampered_report_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = root / "report.json"
            save(report, {"status": "passed"})
            save(root / "receipt.json", {"signature": "same", "report": "report.json", "sha256": study.digest(report)})
            self.assertTrue(study.verified(root, "same"))
            self.assertFalse(study.verified(root, "different"))
            save(report, {"status": "not-established"})
            self.assertFalse(study.verified(root, "same"))

    def test_no_old_runtime_rows_in_new_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save(root / "inputs.json", {"signature": "new"})
            save(root / "r01/receipt.json", {"signature": "old"})
            study.export_csv(root)
            self.assertEqual(len((root / "summary.csv").read_text().splitlines()), 1)

    def test_topology_has_shared_state_and_no_primary_restart(self):
        settings = json.loads((study.ROOT / "settings.json").read_text())
        deployment = document(settings, "sha256:pinned", Path("/owned"), "secret")
        services = deployment["services"]
        self.assertFalse(any(mapping.endswith(":9999") for mapping in services["gateway"]["ports"]))
        for name in ("hub-a", "hub-b"):
            self.assertEqual(services[name]["environment"]["WS_BACKPLANE"], "nats://nats:4222")
            self.assertEqual(services[name]["environment"]["WS_MESSAGE_RETENTION"], "all")
            self.assertNotIn("restart", services[name])
            self.assertEqual(services[name]["image"], "sha256:pinned")


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    """A cleanup problem must never hide the primary workload error or lose the report."""

    async def test_changed_inputs_do_not_overwrite_the_recorded_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            save(output / "inputs.json", {"signature": "previous"})
            original = (output / "inputs.json").read_bytes()
            with (
                patch.object(study, "preflight", return_value={"images": {}, "docker_target": TARGET}),
                patch.object(study, "attempt", side_effect=RuntimeError("unexpected launch")) as attempt,
                self.assertRaisesRegex(RuntimeError, "Inputs changed"),
            ):
                await study.suite({"warmup_seconds": 0, "repetitions": 1}, {}, output)
            self.assertEqual((output / "inputs.json").read_bytes(), original)
            attempt.assert_not_called()

    async def test_unverified_previous_cleanup_blocks_another_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            save(output / "smoke/attempt-01/report.json", {"cleanup_verified": False})
            with (
                patch.object(study, "preflight", return_value={"images": {}, "docker_target": TARGET}),
                patch.object(study, "attempt", side_effect=RuntimeError("unexpected retry")) as attempt,
                self.assertRaisesRegex(RuntimeError, "Unverified cleanup"),
            ):
                await study.suite({"warmup_seconds": 0, "repetitions": 1}, {}, output)
            attempt.assert_not_called()

    async def test_failed_http_upgrades_can_retry_the_same_source_port(self):
        from websockets.exceptions import InvalidStatus

        requests = []

        async def reject(reader, writer):
            try:
                await reader.readuntil(b"\r\n\r\n")
                requests.append(True)
                writer.write(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(reject, "127.0.0.1", 0)
        async with server:
            port = server.sockets[0].getsockname()[1]
            source = available_source_ports(21000, 1, stop=32768)[0]
            for _ in range(3):
                with self.assertRaises(InvalidStatus):
                    await registered(f"ws://127.0.0.1:{port}/ws", "retry-test", source_port=source, reset_on_close=True)
        self.assertEqual(len(requests), 3)

    async def test_failed_start_and_failed_cleanup_still_write_report(self):
        class BrokenDeployment:
            def __init__(self, *_args, **_kwargs):
                self.name = "owned"

            def compose(self, *_args):
                raise RuntimeError("original start failure")

            def cleanup(self):
                raise RuntimeError("cleanup failure")

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "attempt"
            with (
                patch.object(study, "preflight", return_value={"docker_target": TARGET}),
                patch.object(study, "Deployment", BrokenDeployment),
            ):
                report = await study.attempt({}, {"image_id": "pinned"}, output)
            self.assertIn("original start failure", report["error"])
            self.assertIn("cleanup failure", report["cleanup_errors"][0])
            self.assertFalse(report["cleanup_verified"])
            self.assertEqual(json.loads((output / "report.json").read_text())["status"], "not-established")

    async def test_cleanup_attempts_down_after_unpause_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = json.loads((study.ROOT / "settings.json").read_text())
            project = Deployment(settings, "image", Path(folder), docker_target=TARGET)
            calls = []

            def compose(*args):
                calls.append(args[0])
                if args[0] == "unpause":
                    raise RuntimeError("unpause failed")
                return ""

            project.paused = True
            with (
                patch.object(project, "compose", side_effect=compose),
                patch(
                    "topology.command",
                    side_effect=["daemon", "daemon", "", "", "", "daemon"],
                ),
            ):
                errors = project.cleanup()
            self.assertEqual(calls, ["logs", "unpause", "down"])
            self.assertEqual(errors, [])
            self.assertEqual(project.cleanup_diagnostics, ["unpause failed"])
            self.assertFalse((project.directory / "compose.json").exists())

    async def test_fleet_counts_only_live_confirmed_heartbeats(self):
        import time

        fleet = Fleet({"port_base": 18080})
        from types import SimpleNamespace

        from websockets.protocol import State

        fleet.connections[1] = SimpleNamespace(state=State.OPEN)
        fleet.beats = {1: time.monotonic(), 2: time.monotonic()}
        fleet.backends[1] = "b"
        self.assertEqual(fleet.sample()["recent_heartbeats"], 1)
        self.assertEqual(fleet.sample()["on_b"], 1)
        fleet.connections[1].state = State.CLOSED
        self.assertEqual(fleet.sample()["connected"], 0)
        self.assertEqual(fleet.sample()["recent_heartbeats"], 0)
        self.assertEqual(fleet.sample()["on_b"], 0)

    async def test_real_websocket_registration_rejects_reassigned_identity(self):
        from websockets.asyncio.server import serve

        async def hub(socket):
            await socket.recv()
            await socket.send(json.dumps({"type": "et-connect-ack", "agent_id": "wrong"}))
            await socket.wait_closed()

        async with serve(hub, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            with self.assertRaisesRegex(RuntimeError, "identity"):
                await registered(f"ws://127.0.0.1:{port}/ws", "expected")

    async def test_protocol_wait_has_overall_deadline(self):
        class SilentSocket:
            async def recv(self):
                await asyncio.sleep(1)

        with self.assertRaises(TimeoutError):
            await until(SilentSocket(), lambda _: True, timeout=0.01)


if __name__ == "__main__":
    unittest.main()
