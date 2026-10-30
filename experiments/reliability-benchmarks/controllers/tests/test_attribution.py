"""Recovery actors and evidence must distinguish Core supervision from benchmark actions."""

import unittest

from et_experiments.backends.core_recovery import normalize_report
from et_experiments.recovery.attribution import describe_recovery, recovery_mode
from et_experiments.recovery.core import summarize_runs
from et_experiments.utils import BenchmarkError
from recovery_fixtures import core_report


class AttributionTests(unittest.TestCase):
    def test_observe_only_preserves_core_evidence_without_claiming_an_actor(self):
        raw = normalize_report(core_report(status="detected"), "job", "observe-only")
        report = describe_recovery(raw, "observe-only", {"type": "delete_pod", "target": "worker"})
        self.assertFalse(report["incidents"][0]["evidence"]["application_progress_verified"])
        self.assertIsNone(report["metrics"]["mttr_seconds"])
        raw = normalize_report(core_report(), "job", "observe-only")
        report = describe_recovery(raw, "observe-only", {"type": "delete_pod", "target": "worker"}, raw["events"])
        self.assertEqual(report["metrics"]["mttr_seconds"], 8)
        self.assertEqual(report["attribution"]["recovery_actors"], ["unattributed"])
        self.assertFalse(report["attribution"]["intervention_occurred"])
        self.assertTrue(report["incidents"][0]["evidence"]["checkpoint_restoration_verified"])

    def test_assisted_recovery_records_core_not_the_research_controller(self):
        raw = normalize_report(core_report("controller-assisted"), "job", "controller-assisted")
        report = describe_recovery(raw, "controller-assisted", {"type": "none"})
        self.assertEqual(report["attribution"]["recovery_actors"], ["edge-toolkit-core"])
        self.assertTrue(report["attribution"]["intervention_occurred"])
        self.assertFalse(report["attribution"]["core_only_recovery_claim"])

    def test_wall_clock_diagnostics_are_not_compared_with_core_elapsed_time(self):
        raw = normalize_report(core_report(status="detected"), "job", "observe-only")
        event = {"kind": "checkpoint-restored", "workload": "worker", "at": 8}
        report = describe_recovery(raw, "observe-only", {"type": "none"}, [event])
        self.assertNotIn("checkpoint_restored_at", report["incidents"][0]["evidence"])

    def test_requested_rollout_is_not_reported_as_autonomous_replacement(self):
        record = {
            "workload": "worker",
            "status": "recovered",
            "observed_failure_at": 100,
            "recovered_at": 110,
            "recovery_actor": "kubernetes",
            "actions": [],
        }
        report = describe_recovery(
            {"incidents": [record]}, "observe-only", {"type": "rollout_restart", "target": "worker"}
        )
        self.assertEqual(report["attribution"]["recovery_actors"], ["benchmark-requested-kubernetes-recovery"])

    def test_progress_without_actor_evidence_is_not_credited_to_core(self):
        record = {
            "workload": "worker",
            "status": "recovered",
            "observed_failure_at": 100,
            "recovered_at": 110,
            "actions": [],
        }
        report = describe_recovery({"incidents": [record]}, "observe-only", {"type": "network_partition"})
        self.assertEqual(report["attribution"]["recovery_actors"], ["unattributed"])
        self.assertNotIn("connectivity", report["attribution"]["evidence_levels"])

    def test_modes_and_forced_restart_are_validated(self):
        self.assertEqual(recovery_mode({}), "observe-only")
        self.assertEqual(recovery_mode({"recovery_enabled": False}), "disabled")
        for scenario in (
            {"recovery_mode": "unknown"},
            {"require_recovery": True, "recovery_mode": "disabled"},
            {"fault": {"restart_bridge": True}, "recovery_mode": "observe-only"},
        ):
            with self.assertRaises(BenchmarkError):
                recovery_mode(scenario)

    def test_mixed_modes_do_not_produce_an_unqualified_mttr(self):
        results = [
            {
                "attribution": {"recovery_mode": mode},
                "status": "passed",
                "recovery": {
                    "incidents": [{"status": "recovered", "observed_failure_at": 0, "recovered_at": duration}]
                },
            }
            for mode, duration in (("observe-only", 20), ("controller-assisted", 10))
        ]
        summary = summarize_runs(results)
        self.assertTrue(summary["mixed_recovery_modes"])
        self.assertIsNone(summary["mttr_seconds"])
        self.assertEqual(summary["by_recovery_mode"]["observe-only"]["mttr_seconds"], 20)
