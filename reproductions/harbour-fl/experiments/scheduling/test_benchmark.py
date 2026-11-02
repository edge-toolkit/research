import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).with_name("benchmark.py")
SPEC = importlib.util.spec_from_file_location("harbour_fl_scheduling_benchmark", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SchedulingBenchmarkTests(unittest.TestCase):
    def test_expected_scenario_errors_write_rows_and_allow_later_runs(self):
        """Verify expected scenario errors write rows and allow later runs."""
        for error in (
            subprocess.TimeoutExpired("kubectl", 1),
            ValueError("invalid policy"),
            MODULE.CONTROLLER.OrchestrationError("rollout failed"),
        ):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                scenarios = root / "scenarios.json"
                policy = root / "policy.json"
                scenarios.write_text(json.dumps({"one": {}, "two": {}}))
                policy.write_text("{}")
                arguments = [
                    "benchmark",
                    "--all",
                    "--repeat",
                    "2",
                    "--scenarios",
                    str(scenarios),
                    "--policy",
                    str(policy),
                    "--output",
                    str(root),
                ]
                with (
                    patch.object(sys, "argv", arguments),
                    patch.object(
                        MODULE.CONTROLLER, "execute", side_effect=[error, *([{"status": "passed"}] * 3)]
                    ) as execute,
                ):
                    self.assertEqual(MODULE.main(), 1)
                with (root / "scheduling-summary.csv").open() as source:
                    rows = list(csv.DictReader(source))
                self.assertEqual(execute.call_count, 4)
                self.assertEqual([row["status"] for row in rows], ["failed", "passed", "passed", "passed"])
                self.assertEqual(rows[0]["scenario"], "one")
                self.assertIn(str(error), rows[0]["error"])

    def test_scenario_overrides_base_policy_without_description(self):
        result = MODULE.merged_policy(
            {"rounds": 5, "action_mode": "auto"},
            "strict-relocation",
            {"description": "test", "action_mode": "relocate", "allow_same_node_restart": False},
        )
        self.assertEqual(result["scenario"], "strict-relocation")
        self.assertEqual(result["action_mode"], "relocate")
        self.assertNotIn("description", result)

    def test_summary_contains_scheduling_metrics(self):
        row = MODULE.summary_row(
            {
                "scenario": "capacity-rebalance",
                "job_id": "job-1",
                "status": "passed",
                "wall_time_seconds": 20,
                "decision": {"action": "relocate", "source_node": "a", "target_node": "b"},
                "metrics": {
                    "decision_latency_seconds": 0.01,
                    "replacement_ready_seconds": 3.0,
                    "resume_seconds": 8.0,
                    "rounds_lost": 0,
                },
            },
            2,
        )
        self.assertEqual(row["repetition"], 2)
        self.assertEqual(row["target_node"], "b")
        self.assertEqual(row["resume_seconds"], 8.0)
        self.assertEqual(row["wall_time_seconds"], 20)


if __name__ == "__main__":
    unittest.main()
