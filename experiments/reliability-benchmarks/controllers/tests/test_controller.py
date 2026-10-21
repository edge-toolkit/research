import json
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from importlib.resources import files
from pathlib import Path

from et_experiments.web.controller import BenchmarkController, ConfigurationError


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

    def test_defaults_are_applied(self):
        scenario, parameters = self.controller.validate_request({"scenario": "baseline"})
        self.assertEqual(scenario, "baseline")
        self.assertEqual(parameters, {"rounds": 3})

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


if __name__ == "__main__":
    unittest.main()
