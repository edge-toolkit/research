import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from importlib.resources import files
from pathlib import Path
from unittest.mock import Mock, patch

from et_experiments.web.controller import BenchmarkController, ConfigurationError, RunState, main


class ControllerTests(unittest.TestCase):
    def test_ui_keeps_reliability_branding_separate_from_application_title(self):
        page = files("et_experiments.web").joinpath("index.html").read_text(encoding="utf-8")
        self.assertIn("<title>Reliability Benchmark Controller</title>", page)
        self.assertIn("<h1>Reliability Benchmark Controller</h1>", page)
        self.assertIn('<h2 id="title">Application benchmarks</h2>', page)

    def test_external_application_runs_with_controller_owned_outputs(self):
        example = Path(__file__).resolve().parents[2] / "examples/minimal-adapter/catalog.json"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "application"
            shutil.copytree(example.parent, project)
            output = root / "results"
            controller = BenchmarkController(project / "catalog.json", output)
            controller.catalog["adapter"]["command"][0] = sys.executable
            state = controller.start({"scenario": "baseline", "parameters": {"work_units": 2}})
            deadline = time.monotonic() + 10
            while state.status in {"queued", "running"} and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(state.status, "passed", state.logs)
            run = output.resolve() / state.run_id
            self.assertEqual(json.loads((run / "module-result.json").read_text())["completed_work_units"], 2)
            self.assertTrue((run / "result-1.json").is_file())
            request = json.loads((run / "request-1.json").read_text())
            self.assertEqual(request["project"], "Minimal module adapter example")
            self.assertEqual(Path(state.results[0]["report"]).parent, run)

    def test_browser_cannot_override_operator_output_root(self):
        with self.assertRaisesRegex(ConfigurationError, "unsupported parameters"):
            self.controller.validate_request({"scenario": "baseline", "parameters": {"results_dir": "/tmp"}})

    def setUp(self):
        environment = patch.dict(
            os.environ, {k: v for k, v in os.environ.items() if not k.startswith("BENCHMARK_")}, clear=True
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        catalog = {
            "schema_version": 1,
            "adapter": {"command": ["python", "adapter.py"], "cwd": "."},
            "scenarios": {
                "baseline": {
                    "description": "Baseline",
                    "parameters": {"rounds": {"type": "integer", "default": 3, "minimum": 1, "maximum": 10}},
                }
            },
        }
        path = root / "catalog.json"
        path.write_text(json.dumps(catalog), encoding="utf-8")
        self.controller = BenchmarkController(path)
        self.addCleanup(self.controller.shutdown)

    def test_adapter_output_decodes_utf8_and_replaces_invalid_stdout_and_stderr_bytes(self):
        self.controller.catalog["adapter"]["command"] = [
            sys.executable,
            "-c",
            (
                "import os,pathlib; "
                r"os.write(1,b'status \xe2\x9c\x93 \xff\n'); "
                r"os.write(2,b'stderr \xfe\n'); "
                "pathlib.Path(os.environ['BENCHMARK_RESULT_FILE']).write_text('{\"status\":\"passed\"}')"
            ),
        ]
        state = RunState("output-encoding", "baseline", {})
        self.controller._run(state)
        self.assertEqual(state.status, "passed", state.logs)
        self.assertIn("status \u2713 \ufffd", state.logs)
        self.assertIn("stderr \ufffd", state.logs)
        self.assertEqual(state.return_code, 0)

    def test_defaults_are_applied(self):
        scenario, parameters = self.controller.validate_request({"scenario": "baseline"})
        self.assertEqual(scenario, "baseline")
        self.assertEqual(parameters, {"rounds": 3})

    def test_cancel_blocks_new_runs_until_adapter_exits(self):
        started, release = threading.Event(), threading.Event()

        def output():
            started.set()
            yield "adapter running\n"
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release adapter cleanup")

        process = Mock(stdout=output())
        process.poll.return_value = None
        process.wait.return_value = -15
        with patch("et_experiments.web.controller.subprocess.Popen", return_value=process):
            state = self.controller.start({"scenario": "baseline"})
            try:
                self.assertTrue(started.wait(timeout=5))
                self.assertEqual(self.controller.cancel(state.run_id).status, "cancelling")
                self.controller.cancel(state.run_id)
                expected = signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT
                process.send_signal.assert_called_once_with(expected)
                process.terminate.assert_not_called()
                process.wait.assert_not_called()
                with self.assertRaisesRegex(ConfigurationError, "already active"):
                    self.controller.start({"scenario": "baseline"})
                self.assertEqual(state.status, "cancelling")
            finally:
                release.set()
                deadline = time.monotonic() + 5
                while state.status in {"queued", "running", "cancelling"} and time.monotonic() < deadline:
                    time.sleep(0.01)
            self.assertEqual(state.status, "cancelled", state.logs)
            process.wait.assert_called_once()

    def test_cancel_before_worker_starts_remains_cancelled_without_results(self):
        with patch("et_experiments.web.controller.threading.Thread"):
            state = self.controller.start({"scenario": "baseline"})
        self.assertEqual(self.controller.cancel(state.run_id).status, "cancelling")
        with self.assertRaisesRegex(ConfigurationError, "already active"):
            self.controller.start({"scenario": "baseline"})
        with patch("et_experiments.web.controller.subprocess.Popen") as launch:
            self.controller._run(state)
        self.assertEqual(state.status, "cancelled")
        self.assertEqual(state.results, [])
        launch.assert_not_called()

    def test_shutdown_waits_for_live_adapter_cleanup_and_keeps_reading_output(self):
        root = Path(self.temporary.name)
        script = root / "adapter.py"
        script.write_text(
            "import pathlib, signal, time\n"
            "def stop(*args):\n"
            "    raise KeyboardInterrupt\n"
            "signal.signal(signal.SIGINT, stop)\n"
            "if hasattr(signal, 'SIGBREAK'):\n"
            "    signal.signal(signal.SIGBREAK, stop)\n"
            "try:\n"
            "    pathlib.Path('ready').touch()\n"
            "    while True: time.sleep(0.01)\n"
            "except KeyboardInterrupt:\n"
            "    pathlib.Path('cleaning').touch()\n"
            "    deadline = time.monotonic() + 10\n"
            "    while not pathlib.Path('release').exists() and time.monotonic() < deadline: time.sleep(0.01)\n"
            "    print('cleanup completed', flush=True)\n"
        )
        self.controller.catalog["adapter"]["command"] = [sys.executable, str(script)]
        state = self.controller.start({"scenario": "baseline"})
        shutdown = threading.Thread(target=self.controller.shutdown)
        try:
            self.wait_for_file(root / "ready")
            shutdown.start()
            self.wait_for_file(root / "cleaning")
            self.assertEqual(state.status, "cancelling")
            self.assertTrue(shutdown.is_alive())
            with self.assertRaisesRegex(ConfigurationError, "shutting down"):
                self.controller.start({"scenario": "baseline"})
        finally:
            (root / "release").touch()
            if shutdown.ident is not None:
                shutdown.join(timeout=10)
        self.assertFalse(shutdown.is_alive())
        self.assertEqual(state.status, "cancelled", state.logs)
        self.assertIn("cleanup completed", state.logs)
        self.assertTrue(all(not worker.is_alive() for worker in self.controller.workers))

    def wait_for_file(self, path):
        """Wait for a subprocess handshake with a bounded failure time."""
        deadline = time.monotonic() + 5
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(path.exists(), f"adapter did not create {path.name}")

    def test_keyboard_interrupt_shuts_down_workers_before_closing_server(self):
        calls = Mock()
        server = calls.server
        controller = calls.controller
        server.serve_forever.side_effect = KeyboardInterrupt
        with (
            patch("sys.argv", ["et-benchmark-ui", "--catalog", "catalog.json", "serve"]),
            patch("et_experiments.web.controller.BenchmarkController", return_value=controller),
            patch("et_experiments.web.controller.ThreadingHTTPServer", return_value=server),
        ):
            self.assertEqual(main(), 0)
        controller.shutdown.assert_called_once()
        server.server_close.assert_called_once()
        self.assertLess(
            calls.mock_calls.index(unittest.mock.call.controller.shutdown()),
            calls.mock_calls.index(unittest.mock.call.server.server_close()),
        )

    def test_non_object_adapter_results_fail_and_allow_subsequent_runs(self):
        for payload in ("null", "[]", '"invalid"'):
            with self.subTest(payload=payload):
                self.controller.catalog["adapter"]["command"] = [
                    sys.executable,
                    "-c",
                    "import os,pathlib; pathlib.Path(os.environ['BENCHMARK_RESULT_FILE']).write_text("
                    + repr(payload)
                    + ")",
                ]
                state = self.controller.start({"scenario": "baseline"})
                deadline = time.monotonic() + 10
                while state.status in {"queued", "running"} and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(state.status, "failed", state.logs)
                self.assertEqual(state.return_code, 127)
                self.assertTrue(any("AttributeError" in line for line in state.logs))
                self.assertIsNotNone(state.process.poll())

    def test_summary_failure_sets_terminal_state(self):
        state = RunState("summary-failure", "baseline", {})
        process = Mock(stdout=io.StringIO())
        process.wait.return_value = 0
        process.poll.return_value = 0
        with (
            patch("et_experiments.web.controller.subprocess.Popen", return_value=process),
            patch("et_experiments.web.controller.summarize_runs", side_effect=KeyError("recovered_at")),
        ):
            self.controller._run(state)
        self.assertEqual(state.status, "failed")
        self.assertTrue(any("KeyError" in line for line in state.logs))
        process.terminate.assert_not_called()

    def test_unexpected_stream_failure_terminates_live_adapter(self):
        for stubborn in (False, True):
            with self.subTest(stubborn=stubborn):
                state = RunState(f"stream-failure-{stubborn}", "baseline", {})
                stream = Mock()
                stream.__iter__ = Mock(side_effect=RuntimeError("stream failed"))
                process = Mock(stdout=stream)
                process.poll.return_value = None
                if stubborn:
                    process.wait.side_effect = [subprocess.TimeoutExpired("adapter", 10), 0]
                with patch("et_experiments.web.controller.subprocess.Popen", return_value=process):
                    self.controller._run(state)
                self.assertEqual(state.status, "failed")
                self.assertEqual(state.return_code, 127)
                self.assertTrue(any("RuntimeError" in line for line in state.logs))
                process.terminate.assert_called_once()
                self.assertEqual(process.kill.call_count, int(stubborn))
                self.assertEqual(process.wait.call_count, 2 if stubborn else 1)

    def test_unknown_parameters_are_rejected(self):
        with self.assertRaisesRegex(ConfigurationError, "unsupported parameters"):
            self.controller.validate_request({"scenario": "baseline", "parameters": {"command": "kubectl"}})

    def test_out_of_range_parameters_are_rejected(self):
        with self.assertRaisesRegex(ConfigurationError, "above its maximum"):
            self.controller.validate_request({"scenario": "baseline", "parameters": {"rounds": 100}})

    def test_running_process_can_be_serialized_and_repetitions_are_summarized(self):
        self.controller.catalog["parameters"] = {
            "repetitions": {"type": "integer", "default": 2, "minimum": 1, "maximum": 3}
        }
        self.controller.catalog["adapter"]["command"] = [
            sys.executable,
            "-c",
            (
                "import os,json,pathlib; "
                "pathlib.Path(os.environ['BENCHMARK_RESULT_FILE']).write_text(json.dumps({'status':'passed'}))"
            ),
        ]
        state = self.controller.start({"scenario": "baseline"})
        deadline = time.monotonic() + 10
        while state.status in {"queued", "running"} and time.monotonic() < deadline:
            json.dumps(state.public())
            time.sleep(0.01)
        self.assertEqual(state.status, "passed", state.logs)
        self.assertEqual(state.summary["repetitions"], 2)
        self.assertIsNone(state.summary["mttr_seconds"])
        self.assertEqual(state.summary["experiment_error_ratio"], 0)

    def test_public_state_excludes_process_without_deepcopy(self):
        from et_experiments.web.controller import RunState

        process = subprocess.Popen([sys.executable, "-c", "pass"])
        try:
            state = RunState("test", "baseline", {}, process=process)
            self.assertNotIn("process", state.public())
        finally:
            process.wait(timeout=10)

    def test_nonfinite_adapter_results_and_derived_summary_write_standard_json(self):
        value = {
            "status": "passed",
            "sli": {"loss": float("nan")},
            "recovery": {
                "incidents": [
                    {"status": "recovered", "observed_failure_at": 0, "detected_at": 1, "recovered_at": float("inf")}
                ]
            },
        }
        self.controller.catalog["adapter"]["command"] = [
            sys.executable,
            "-c",
            "import os,pathlib; pathlib.Path(os.environ['BENCHMARK_RESULT_FILE']).write_text("
            + repr(json.dumps(value))
            + ")",
        ]
        state = self.controller.start({"scenario": "baseline"})
        deadline = time.monotonic() + 10
        while state.status in {"queued", "running"} and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(state.status, "passed", state.logs)
        run = self.controller.results_root / state.run_id
        for path in (run / "result-1.json", run / "summary.json"):
            result = json.loads(path.read_text(), parse_constant=lambda token: self.fail(token))
            if path.name == "result-1.json":
                self.assertIsNone(result["sli"]["loss"])
                self.assertIsNone(result["recovery"]["incidents"][0]["recovered_at"])
            else:
                self.assertIsNone(result["mttr_seconds"])


if __name__ == "__main__":
    unittest.main()
