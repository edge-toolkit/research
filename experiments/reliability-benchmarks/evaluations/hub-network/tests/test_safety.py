"""Protect the host and preserve the identity and validity of network measurements."""

import contextlib
import io
import json
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import diagnostics
import monitor
import study
from network import topology


class SafetyTests(unittest.TestCase):
    def test_resume_rejects_damaged_receipts_and_evidence_outside_the_case(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            case = root / "baseline-r01"
            path = case / "attempt-01/report.json"
            report = {"cleanup_verified": True, "measurement": {"target_held": True}}
            study.save(path, report)
            receipt = {"signature": "same", "report": str(path.relative_to(root)), "sha256": study.digest(path)}
            study.save(case / "receipt.json", receipt)
            self.assertEqual(study.receipt_report(case, root, "same"), report)
            self.assertIsNone(study.receipt_report(case, root, "different"))
            for content in ("{", "[]", '{"signature":"same","report":null}'):
                (case / "receipt.json").write_text(content)
                self.assertIsNone(study.receipt_report(case, root, "same"))
            outside = root / "external.json"
            study.save(outside, report)
            study.save(case / "receipt.json", {**receipt, "report": "external.json", "sha256": study.digest(outside)})
            self.assertIsNone(study.receipt_report(case, root, "same"))
            study.save(case / "receipt.json", receipt)
            path.write_text("damaged evidence")
            self.assertIsNone(study.receipt_report(case, root, "same"))

    def test_unstarted_watchdog_can_be_closed_during_failed_attempt_cleanup(self):
        watcher = study.ProtectionWatch(lambda: None)
        with (
            patch.object(watcher.thread, "start", side_effect=RuntimeError("cannot start new thread")),
            self.assertRaisesRegex(RuntimeError, "cannot start new thread"),
        ):
            watcher.thread.start()
        watcher.close()

    def test_slow_metrics_cannot_delay_a_host_protection_stop(self):
        metrics_started, stopped = threading.Event(), threading.Event()
        process = Mock(returncode=0)
        process.poll.return_value = None
        reads = 0

        def compose(_project, _file, *args, **_kwargs):
            nonlocal reads
            if "--host" in args:
                reads += 1
                return SimpleNamespace(
                    stdout=json.dumps(
                        {
                            "boot_id": "host",
                            "cpu_total": reads * 100,
                            "cpu_idle": reads * 90,
                            "swap_out_bytes": 0,
                            "available_memory_bytes": 0 if metrics_started.is_set() else 64 * 2**30,
                        }
                    )
                )
            if args[0] == "stop":
                process.poll.return_value = 0
                stopped.set()
            return SimpleNamespace(returncode=0, stdout="{}", stderr="")

        def command(args, **_kwargs):
            if args[:2] == ["docker", "stats"]:
                metrics_started.set()
                if not stopped.wait(4):
                    raise TimeoutError("Protection did not run while metrics were blocked")
            return SimpleNamespace(returncode=0, stdout="daemon" if args[1] == "info" else "", stderr="")

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(study, "compose", side_effect=compose),
            patch.object(study, "command", side_effect=command),
            patch.object(
                study,
                "setup",
                return_value={
                    "hub_address": "hub",
                    "interfaces": {"rate": [], "delay": []},
                },
            ),
            patch.object(study.subprocess, "Popen", return_value=process),
            patch.object(study.time, "sleep"),
        ):
            report = study.execute(
                {"id": "baseline"},
                study.configuration(),
                {"id": "image"},
                Path(folder) / "attempt",
            )
        self.assertTrue(stopped.is_set())
        self.assertEqual(report["outcome"], "protection:host-memory-reserve")
        self.assertTrue(report["cleanup_verified"])

    def test_explicit_cleanup_timeout_still_checks_the_owned_inventory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            attempt = root / "baseline/attempt-01"
            study.save(
                attempt / "project.json",
                {"daemon_id": "daemon", "project": "et-hub-network-0123456789"},
            )
            with (
                patch.object(study, "command", return_value=SimpleNamespace(stdout="daemon")),
                patch.object(study, "compose", side_effect=subprocess.TimeoutExpired("down", 120)),
                patch.object(study, "owned_resources") as inventory,
            ):
                study.cleanup_attempts(root)
            inventory.assert_called_once()
            report = json.loads((attempt / "report.json").read_text())
            self.assertTrue(report["cleanup_verified"])
            self.assertIn("timed out", report["cleanup_diagnostics"][0])

    def test_changed_daemon_cannot_receive_teardown_or_prove_absence(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(
                study,
                "command",
                side_effect=[
                    SimpleNamespace(stdout="original"),
                    SimpleNamespace(stdout="different"),
                    SimpleNamespace(stdout="different"),
                ],
            ),
            patch.object(study, "compose", side_effect=RuntimeError("startup failed")) as compose,
        ):
            report = study.execute(
                {"id": "baseline"},
                study.configuration(),
                {"id": "image"},
                Path(folder) / "attempt",
            )
        self.assertFalse(report["cleanup_verified"])
        self.assertFalse(any(call.args[2] == "down" for call in compose.call_args_list))

    def test_directory_sync_failure_prevents_any_deployment(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(study, "command", return_value=SimpleNamespace(stdout="daemon")),
            patch.object(study, "sync_directory", side_effect=OSError("sync failed")),
            patch.object(study, "compose") as compose,
            self.assertRaisesRegex(OSError, "sync failed"),
        ):
            study.execute(
                {"id": "baseline"},
                {},
                {"id": "image"},
                Path(folder) / "new/baseline/attempt-01",
            )
        compose.assert_not_called()

    def test_cohort_request_ignores_stale_observations(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            study.save(
                root / "cohort-observation.json",
                {"request": "old", "cohort": {"connected": 25000}},
            )

            def respond(_seconds):
                request = json.loads((root / "cohort-request.json").read_text())["request"]
                study.save(
                    root / "cohort-observation.json",
                    {"request": request, "cohort": {"connected": 24999}},
                )

            with patch.object(study.time, "sleep", side_effect=respond):
                self.assertEqual(study.observe_cohort(root)["connected"], 24999)

    def test_low_memory_is_rechecked_before_compose_starts(self):
        self.host_sensor.return_value = {"available_memory_bytes": 0}
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(study, "command", return_value=SimpleNamespace(stdout="daemon")),
            patch.object(study, "compose") as compose,
        ):
            report = study.execute({"id": "baseline"}, study.configuration(), {"id": "image"}, Path(folder) / "attempt")
            self.assertEqual(report["outcome"], "protection:host-memory-reserve")
            self.assertTrue(report["cleanup_verified"])
            compose.assert_not_called()

    def test_failed_host_sensor_stops_live_work_before_log_collection(self):
        state = {"reads": 0}
        actions = []
        process = Mock(returncode=None)
        process.poll.return_value = None

        def compose(_project, _file, *args, **_kwargs):
            actions.append(args[0])
            if "--host" in args:
                state["reads"] += 1
                if state["reads"] == 4:
                    raise subprocess.TimeoutExpired("host probe", 15)
                return SimpleNamespace(
                    stdout=json.dumps(
                        {
                            "boot_id": "host",
                            "cpu_total": state["reads"] * 100,
                            "cpu_idle": state["reads"] * 90,
                            "swap_out_bytes": 0,
                            "available_memory_bytes": 64 * 2**30,
                        }
                    )
                )
            return SimpleNamespace(returncode=0, stdout="{}", stderr="")

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(study, "compose", side_effect=compose),
            patch.object(
                study,
                "command",
                side_effect=lambda args, **_kw: SimpleNamespace(stdout="daemon" if args[1] == "info" else ""),
            ),
            patch.object(study, "setup", return_value={"hub_address": "hub", "interfaces": {"rate": [], "delay": []}}),
            patch.object(study.subprocess, "Popen", return_value=process),
            patch.object(study, "json_lines", return_value=[]),
        ):
            report = study.execute({"id": "baseline"}, study.configuration(), {"id": "image"}, Path(folder) / "attempt")
        self.assertIn("TimeoutExpired", report["host_sampling_error"])
        self.assertIn("host probe", report["harness_error"])
        self.assertLess(actions.index("stop"), actions.index("logs"))
        self.assertTrue(report["cleanup_verified"])

    def test_source_edits_after_image_verification_cannot_be_saved_as_matching_inputs(self):
        from types import SimpleNamespace

        image = {"build_inputs": study.build_identity()}
        original = study.digest

        def changed(path):
            return "0" * 64 if str(path).endswith("driver/src/main.rs") else original(path)

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(study, "image_identity", return_value=image),
            patch.object(study, "digest", side_effect=changed),
            patch.object(study, "execute") as execute,
        ):
            args = SimpleNamespace(command="smoke", agents=1, case="baseline", output=Path(folder))
            with self.assertRaisesRegex(RuntimeError, "Captured helper build inputs differ"):
                study.run(args)
            self.assertFalse((Path(folder) / "inputs.json").exists())
            execute.assert_not_called()

    def setUp(self):
        patcher = patch.object(
            study,
            "workload_host",
            return_value={"available_memory_bytes": 64 * 2**30, "total_memory_bytes": 128 * 2**30},
        )
        self.host_sensor = patcher.start()
        self.addCleanup(patcher.stop)

    def test_duplicate_nonbaseline_ids_are_rejected(self):
        config = study.configuration()
        config["cases"].append({**config["cases"][1], "delay_ms_each_direction": 999})
        with self.assertRaisesRegex(ValueError, "case ID must be unique"):
            study.selected_cases(config, None)

    def test_protection_checks_setup_probe_and_final_measurement_boundaries(self):
        for boundary in ("after-setup", "after-probe", "finished"):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as folder:
                state = {"reads": 0, "probe": False}
                threshold = 3 if boundary == "after-setup" else 4
                process = Mock(returncode=0)
                process.poll.return_value = 0

                def compose(_project, _file, *args, state=state, threshold=threshold, **_kwargs):
                    if "--host" in args:
                        state["reads"] += 1
                        return SimpleNamespace(
                            stdout=json.dumps(
                                {
                                    "boot_id": "host",
                                    "cpu_total": state["reads"] * 100,
                                    "cpu_idle": state["reads"] * 90,
                                    "swap_out_bytes": 0,
                                    "available_memory_bytes": 0 if state["reads"] >= threshold else 64 * 2**30,
                                }
                            )
                        )
                    if "--verify-bytes" in args:
                        state["probe"] = True
                        return SimpleNamespace(stdout=json.dumps({"seconds": 10}))
                    return SimpleNamespace(returncode=0, stdout="{}", stderr="")

                with (
                    patch.object(study, "compose", side_effect=compose),
                    patch.object(
                        study,
                        "command",
                        side_effect=lambda args, **_kwargs: SimpleNamespace(
                            returncode=0, stdout="daemon" if args[1] == "info" else "", stderr=""
                        ),
                    ),
                    patch.object(
                        study,
                        "setup",
                        return_value={"hub_address": "hub:8080", "interfaces": {"rate": [], "delay": []}},
                    ),
                    patch.object(study.subprocess, "Popen", return_value=process) as start,
                    patch.object(study, "json_lines", return_value=[{"type": "report", "target_held": True}]),
                ):
                    case = {"id": "bandwidth-10kbit-per-vessel" if boundary == "after-probe" else "baseline"}
                    report = study.execute(case, study.configuration(), {"id": "image"}, Path(folder) / "attempt")
                self.assertEqual(report["outcome"], "protection:host-memory-reserve")
                self.assertEqual(state["probe"], boundary == "after-probe")
                self.assertEqual(start.call_count, int(boundary == "finished"))
                self.assertTrue(report["cleanup_verified"])

    def test_check_rejects_low_initial_memory_without_starting_compose(self):
        import sys

        config = study.configuration()
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(sys, "argv", ["study", "check", "--output", folder]),
            patch.object(study, "ROOT", Path(folder)),
            patch.object(study, "image_identity", return_value={"id": "image"}),
            patch.object(study, "command", return_value=SimpleNamespace(stdout="")),
            patch.object(
                study, "workload_host", return_value={"total_memory_bytes": 64 * 2**30, "available_memory_bytes": 1}
            ),
            patch.object(study, "execute") as execute,
        ):
            study.save(Path(folder) / "settings.json", config)
            with self.assertRaisesRegex(RuntimeError, "Insufficient available"):
                study.main()
            execute.assert_not_called()

    def test_changed_build_inputs_require_rebuilding_before_image_inspection(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(study, "ROOT", Path(folder)):
            study.save(Path(folder) / ".runtime/image.json", {"build_inputs": "old"})
            with patch.object(study, "build_identity", return_value="new"), patch.object(study, "command") as command:
                with self.assertRaisesRegex(RuntimeError, "Helper build inputs changed"):
                    study.image_identity()
                command.assert_not_called()

    def test_baseline_is_first_even_when_settings_are_reordered(self):
        config = study.configuration()
        config["cases"].reverse()
        cases = study.selected_cases(config, "latency-500ms-rtt")
        self.assertEqual([case["id"] for case in cases], ["baseline", "latency-500ms-rtt"])
        config["cases"] = cases[1:]
        with self.assertRaisesRegex(ValueError, "baseline"):
            study.selected_cases(config, None)

    def test_rejected_baseline_has_no_receipt_and_can_be_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            passed = False

            def execute(_case, _config, _image, directory):
                result = {"cleanup_verified": True, "outcome": "recorded", "measurement": {"target_held": passed}}
                study.save(directory / "report.json", result)
                return result

            args = SimpleNamespace(command="run", agents=None, output=output, case="baseline")
            with (
                patch.object(study, "image_identity", return_value={"build_inputs": study.build_identity()}),
                patch.object(study.platform, "system", return_value="Linux"),
                patch.object(
                    study,
                    "workload_host",
                    return_value={"total_memory_bytes": 64 * 2**30, "available_memory_bytes": 32 * 2**30},
                ),
                patch.object(study, "execute", side_effect=execute),
            ):
                with self.assertRaisesRegex(RuntimeError, "Baseline did not maintain"):
                    study.run(args)
                self.assertFalse((output / "baseline-r01/receipt.json").exists())
                passed = True
                study.run(args)
                self.assertTrue((output / "baseline-r01/attempt-02/report.json").is_file())
                self.assertTrue((output / "baseline-r01/receipt.json").is_file())

    def test_owned_network_must_be_removed_even_if_containers_are_gone(self):
        with (
            patch.object(
                study, "command", side_effect=[SimpleNamespace(stdout=""), SimpleNamespace(stdout="underlay")]
            ),
            self.assertRaisesRegex(RuntimeError, "Owned networks remain"),
        ):
            study.owned_resources("owned")

    def test_cleanup_is_bound_to_daemon_and_reverified_after_manual_removal(self):
        for daemon in ("original", "different"):
            with self.subTest(daemon=daemon), tempfile.TemporaryDirectory() as folder:
                output = Path(folder)
                attempt = output / "baseline-r01/attempt-01"
                completed = output / "completed/attempt-01/report.json"
                study.save(completed, {"cleanup_verified": True, "outcome": "target-held"})
                original = completed.read_bytes()
                study.save(attempt / "project.json", {"project": "et-hub-network-0123456789", "daemon_id": "original"})
                study.save(attempt / "report.json", {"cleanup_verified": False})
                with (
                    patch.object(study, "command", return_value=SimpleNamespace(stdout=daemon)),
                    patch.object(study, "compose", return_value=SimpleNamespace(returncode=0)) as compose,
                    patch.object(study, "owned_resources") as inventory,
                ):
                    if daemon == "different":
                        with self.assertRaisesRegex(RuntimeError, "target differs"):
                            study.cleanup_attempts(output)
                        compose.assert_not_called()
                    else:
                        study.cleanup_attempts(output)
                        inventory.assert_called_once_with("et-hub-network-0123456789")
                        self.assertTrue(json.loads((attempt / "report.json").read_text())["cleanup_verified"])
                        self.assertEqual(completed.read_bytes(), original)

    def test_colon_in_results_path_is_a_structured_bind_source(self):
        source = Path("/tmp/run:trial")
        mounts = topology("image", source, study.configuration())["services"]["driver"]["volumes"]
        self.assertEqual(mounts, [{"type": "bind", "source": str(source.resolve()), "target": "/results"}])

    def test_compose_bind_escapes_literal_dollar_signs(self):
        mounts = topology("image", Path("/tmp/trial$RUN"), study.configuration())["services"]["driver"]["volumes"]
        self.assertEqual(mounts[0]["source"], str(Path("/tmp/trial$RUN").resolve()).replace("$", "$$"))

    def test_connectivity_timeout_retains_stage_and_partial_output(self):
        from network import connectivity

        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(subprocess.TimeoutExpired):
                connectivity(
                    Mock(side_effect=subprocess.TimeoutExpired("exec", 15, output=b"partial")), Path(folder), "ready"
                )
            row = json.loads((Path(folder) / "connectivity-ready.json").read_text())
            self.assertEqual(row["stdout"], "partial")
            self.assertEqual(row["stage"], "ready")

    def test_output_lock_is_shared_by_resolved_path_and_released(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "evidence"
            with (
                study.output_lock(output),
                self.assertRaises(BlockingIOError),
                study.output_lock(output / ".." / "evidence"),
            ):
                self.fail("second lock acquired")
            with study.output_lock(output):
                pass

    def test_unstarted_attempt_cleanup_needs_no_compose(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            study.save(output / "preflight/attempt-01/project.json", {"phase": "not-started", "daemon_id": "daemon"})
            with (
                patch.object(study, "command", return_value=SimpleNamespace(stdout="daemon")),
                patch.object(study, "compose") as compose,
            ):
                study.cleanup_attempts(output)
                compose.assert_not_called()
            self.assertTrue(json.loads((output / "preflight/attempt-01/report.json").read_text())["cleanup_verified"])

    def test_missing_cgroup_v2_fails_before_reading_accounting(self):
        with patch.object(monitor.Path, "is_file", return_value=False), patch.object(monitor, "pairs") as pairs:
            with self.assertRaisesRegex(RuntimeError, "requires cgroup v2"):
                monitor.snapshot()
            pairs.assert_not_called()

    def test_diagnostic_timeout_retains_completed_probes(self):
        results = [SimpleNamespace(returncode=0, stdout="address-data", stderr=""), subprocess.TimeoutExpired("ip", 2)]
        results += [SimpleNamespace(returncode=0, stdout="remaining", stderr="")] * 3
        out = io.StringIO()
        with (
            patch.object(diagnostics.subprocess, "run", side_effect=results),
            patch.object(diagnostics.Path, "read_text", return_value=""),
            patch.object(diagnostics.Path, "glob", return_value=[]),
            contextlib.redirect_stdout(out),
        ):
            diagnostics.main()
        report = json.loads(out.getvalue())
        self.assertEqual(report["addresses"]["stdout"], "address-data")
        self.assertIn("timed out", report["links"]["error"])
        self.assertEqual(report["tunnels"]["stdout"], "remaining")

    def test_failed_proc_reads_do_not_discard_completed_diagnostics(self):
        out = io.StringIO()
        with (
            patch.object(
                diagnostics.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="collected", stderr="")
            ),
            patch.object(diagnostics.Path, "read_text", side_effect=FileNotFoundError("removed interface")),
            patch.object(diagnostics.Path, "glob", return_value=[Path("/proc/missing/rp_filter")]),
            contextlib.redirect_stdout(out),
        ):
            diagnostics.main()
        report = json.loads(out.getvalue())
        self.assertEqual(report["routes"]["stdout"], "collected")
        self.assertIn("error", report["tcp"])
        self.assertIn("error", report["sysctls"]["/proc/missing/rp_filter"])

    def test_guard_and_cohort_checks_withhold_the_impairment_gate(self):
        for scenario in ("before-change", "after-change", "cohort-loss"):
            with (
                self.subTest(scenario=scenario),
                tempfile.TemporaryDirectory() as folder,
            ):
                output = Path(folder) / "attempt"
                command_calls = []
                process = Mock()
                process.poll.return_value = None
                state = {"host_reads": 0, "changed": False}

                def compose(_project, _file, *args, state=state, scenario=scenario, **_kwargs):
                    if "change" in args:
                        state["changed"] = True
                    if "--host" in args:
                        state["host_reads"] += 1
                        host_reads = state["host_reads"]
                        breach = (scenario == "before-change" and host_reads == 4) or (
                            scenario == "after-change" and host_reads == 5
                        )
                        return SimpleNamespace(
                            stdout=json.dumps(
                                {
                                    "boot_id": "boot",
                                    "cpu_total": host_reads * 100,
                                    "cpu_idle": host_reads * 90,
                                    "available_memory_bytes": 0 if breach else 64 * 2**30,
                                    "swap_out_bytes": 0,
                                }
                            )
                        )
                    return SimpleNamespace(returncode=0, stdout="{}", stderr="")

                def rows(_path, state=state, scenario=scenario):
                    return [
                        {
                            "phase": "awaiting-impairment",
                            "connected": 24999 if state["changed"] and scenario == "cohort-loss" else 25000,
                        }
                    ]

                def command(args, command_calls=command_calls, **_kwargs):
                    command_calls.append(args)
                    return SimpleNamespace(returncode=0, stdout="daemon" if args[1] == "info" else "", stderr="")

                with (
                    patch.object(study, "compose", side_effect=compose),
                    patch.object(study, "command", side_effect=command),
                    patch.object(
                        study,
                        "setup",
                        return_value={
                            "hub_address": "hub:8080",
                            "interfaces": {"rate": [], "delay": []},
                        },
                    ),
                    patch.object(study.subprocess, "Popen", return_value=process),
                    patch.object(study, "json_lines", side_effect=rows),
                    patch.object(
                        study,
                        "observe_cohort",
                        side_effect=lambda _directory: rows(None)[-1],
                    ),
                ):
                    config = study.configuration()
                    case = next(case for case in config["cases"] if case["id"] == "loss-30pct")
                    report = study.execute(case, config, {"id": "image"}, output)
                self.assertFalse((output / "impairment-applied.json").exists())
                self.assertFalse(any(args[:2] == ["docker", "stats"] for args in command_calls))
                self.assertEqual(state["changed"], scenario != "before-change")
                if scenario == "cohort-loss":
                    self.assertIn("Cohort became incomplete", report["harness_error"])
                else:
                    self.assertEqual(report["outcome"], "protection:host-memory-reserve")

    def test_check_selects_baseline_and_rejects_harness_errors(self):
        import sys

        with tempfile.TemporaryDirectory() as folder:
            config = study.configuration()
            config["cases"].reverse()
            with (
                patch.object(sys, "argv", ["study", "check", "--output", folder]),
                patch.object(study, "ROOT", Path(folder)),
                patch.object(study, "image_identity", return_value={"id": "image"}),
                patch.object(study, "command", return_value=SimpleNamespace(stdout="")),
                patch.object(
                    study,
                    "workload_host",
                    return_value={"total_memory_bytes": 64 * 2**30, "available_memory_bytes": 32 * 2**30},
                ),
                patch.object(
                    study,
                    "execute",
                    return_value={
                        "outcome": "target-held",
                        "cleanup_verified": True,
                        "measurement": {"target_held": True},
                        "harness_error": "logs failed",
                    },
                ) as execute,
            ):
                study.save(Path(folder) / "settings.json", config)
                with self.assertRaisesRegex(RuntimeError, "preflight failed"):
                    study.main()
                self.assertEqual(execute.call_args.args[0]["id"], "baseline")
                self.assertEqual(execute.call_args.args[3], Path(folder).resolve() / "preflight/attempt-01")

    def test_default_check_blocks_a_previous_unverified_preflight_before_docker_calls(self):
        import sys

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            study.save(root / "results/check/preflight/attempt-01/report.json", {"cleanup_verified": False})
            with (
                patch.object(sys, "argv", ["study", "check"]),
                patch.object(study, "ROOT", root),
                patch.object(study, "command") as command,
                patch.object(study, "image_identity") as image,
            ):
                with self.assertRaisesRegex(RuntimeError, "Unverified cleanup"):
                    study.main()
                command.assert_not_called()
                image.assert_not_called()

    def test_host_memory_guard_stops_work_before_optional_metrics(self):
        calls = []
        command_calls = []
        host_reads = 0
        process = Mock()
        process.poll.return_value = None

        def command(args, **_kwargs):
            command_calls.append(args)
            return SimpleNamespace(returncode=0, stdout="daemon" if args[:2] == ["docker", "info"] else "", stderr="")

        def compose(_project, _file, *args, **_kwargs):
            nonlocal host_reads
            calls.append(args)
            if "--host" in args:
                host_reads += 1
                return SimpleNamespace(
                    stdout=json.dumps(
                        {
                            "boot_id": "boot",
                            "total_memory_bytes": 64 * 2**30,
                            "available_memory_bytes": 32 * 2**30 if host_reads < 4 else 0,
                            "cpu_total": host_reads * 100,
                            "cpu_idle": host_reads * 90,
                            "swap_out_bytes": 0,
                        }
                    )
                )
            return SimpleNamespace(returncode=0, stdout="{}" if "monitor.py" in " ".join(args) else "id", stderr="")

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(study, "command", side_effect=command),
            patch.object(study, "compose", side_effect=compose),
            patch.object(
                study, "setup", return_value={"hub_address": "hub:8080", "interfaces": {"rate": [], "delay": []}}
            ),
            patch.object(study.subprocess, "Popen", return_value=process),
            patch.object(study, "json_lines", return_value=[]),
        ):
            report = study.execute({"id": "baseline"}, study.configuration(), {"id": "image"}, Path(folder) / "attempt")
        self.assertEqual(report["outcome"], "protection:host-memory-reserve")
        self.assertFalse(any(args[:2] == ["docker", "stats"] for args in command_calls))
        process.terminate.assert_called_once()
        self.assertIn(("stop", "--timeout", "1", "driver", "hub"), calls)
        self.assertTrue(report["cleanup_verified"])

    def test_final_measurement_does_not_hide_a_failed_driver_exit(self):
        for code in (0, 1):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as folder:
                process = Mock(returncode=code)
                process.poll.return_value = code
                host = {
                    "boot_id": "boot",
                    "cpu_total": 100,
                    "cpu_idle": 90,
                    "available_memory_bytes": 64 * 2**30,
                    "swap_out_bytes": 0,
                }

                def compose(_project, _file, *args, host=host, **_kwargs):
                    value = json.dumps(host) if "--host" in args else "{}"
                    return SimpleNamespace(returncode=0, stdout=value, stderr="")

                def command(args, **_kwargs):
                    return SimpleNamespace(stdout="daemon" if args[1] == "info" else "", returncode=0, stderr="")

                with (
                    patch.object(study, "compose", side_effect=compose),
                    patch.object(study, "command", side_effect=command),
                    patch.object(
                        study,
                        "setup",
                        return_value={"hub_address": "hub:8080", "interfaces": {"rate": [], "delay": []}},
                    ),
                    patch.object(study.subprocess, "Popen", return_value=process),
                    patch.object(study, "json_lines", return_value=[{"type": "report", "target_held": True}]),
                ):
                    report = study.execute(
                        {"id": "baseline"}, study.configuration(), {"id": "image"}, Path(folder) / "attempt"
                    )
                self.assertTrue(report["cleanup_verified"])
                if code:
                    self.assertIn("Driver exited 1", report["harness_error"])
                    self.assertNotEqual(report["outcome"], "target-held")
                else:
                    self.assertEqual(report["outcome"], "target-held")
