"""Graceful supervisor replacement must retain strict durable continuity guards."""

import copy
import unittest

from et_experiments.backends.core_recovery import normalize_report
from et_experiments.recovery.supervision import verify_continuation
from et_experiments.utils import BenchmarkError


def reports():
    """Build a same-second replacement with a consumed restart budget."""
    before = {
        "schema_version": 1,
        "job_id": "job",
        "mode": "controller-assisted",
        "clock_basis": "observer-monotonic-seconds",
        "context": {
            "schema_version": 1,
            "configuration": {"job_id": "job", "policy": {"max_restarts": 1}},
            "jitter_seed": 13234799103983388871,
            "elapsed_seconds": 206,
            "restored_sessions": 0,
        },
        "observation_gaps": [{"started_at": 0, "ended_at": 0, "reason": "hub registration"}],
        "restart_budget": {"restarts": {"vessel-3": 1}},
        "incidents": [
            {
                "workload": "vessel-3",
                "onset": 138,
                "detected": 142,
                "recovered": None,
                "baseline": 1,
                "status": "detected",
                "actions": [{"kind": "restart-reserved"}, {"kind": "restart-failed", "error": "forbidden"}],
                "stability": {"status": "disabled"},
            }
        ],
        "checkpoint_measurements": [],
        "mttr_seconds": None,
        "mean_failure_detection_seconds": 4.0,
        "unknown_observations": 0,
    }
    after = copy.deepcopy(before)
    after["clock_basis"] = "observer-elapsed-seconds-with-restart-gaps"
    after["context"]["restored_sessions"] = 1
    after["observation_gaps"] += [
        {"started_at": 206, "ended_at": 206, "reason": "supervision stopped"},
        {"started_at": 206, "ended_at": 206, "reason": "hub registration"},
    ]
    return before, after


class GracefulRestorationTests(unittest.TestCase):
    def test_deployed_same_second_graceful_replacement_preserves_raw_evidence(self):
        before, after = reports()
        result = normalize_report(after, "job", "controller-assisted")
        verify_continuation(before, after)
        self.assertEqual(result["restored_sessions"], 1)
        self.assertEqual(result["core_report"], after)
        self.assertEqual(result["metrics"]["recovery_success_ratio"], 0)
        self.assertEqual(result["metrics"]["recovery_attempts"], 1)
        self.assertEqual(result["observation_gaps"][1]["reason"], "supervision stopped")

    def test_explicit_restart_reason_remains_supported(self):
        before, after = reports()
        after["observation_gaps"][1]["reason"] = "supervisor restart"
        normalize_report(after, "job", "controller-assisted")
        verify_continuation(before, after)

    def test_stop_label_does_not_replace_context_validation(self):
        for invalid in ("schema", "sessions", "job"):
            _, after = reports()
            if invalid == "schema":
                after["context"]["schema_version"] = 2
            elif invalid == "sessions":
                after["context"]["restored_sessions"] = 0
            else:
                after["context"]["configuration"]["job_id"] = "foreign"
            with self.subTest(invalid=invalid), self.assertRaises(BenchmarkError):
                normalize_report(after, "job", "controller-assisted")

    def test_unrelated_open_future_or_stale_gap_does_not_establish_replacement(self):
        for invalid in ("reason", "open", "future", "backwards", "stale"):
            before, after = reports()
            gap = after["observation_gaps"][1]
            if invalid == "reason":
                gap["reason"] = "hub registration"
            elif invalid == "open":
                gap["ended_at"] = None
            elif invalid == "future":
                gap["ended_at"] = 207
            elif invalid == "backwards":
                gap["started_at"] = 207
            else:
                gap.update(started_at=100, ended_at=101)
            with self.subTest(invalid=invalid), self.assertRaises(BenchmarkError):
                verify_continuation(before, after)

    def test_replacement_still_requires_seed_history_configuration_and_budget(self):
        for invalid in ("seed", "configuration", "sessions", "clock", "actions", "budget", "checkpoints"):
            before, after = reports()
            if invalid == "seed":
                after["context"]["jitter_seed"] += 1
            elif invalid == "configuration":
                after["context"]["configuration"]["policy"]["max_restarts"] = 2
            elif invalid == "sessions":
                after["context"]["restored_sessions"] = 0
            elif invalid == "clock":
                after["context"]["elapsed_seconds"] = 205
            elif invalid == "actions":
                after["incidents"][0]["actions"] = []
            elif invalid == "budget":
                after["restart_budget"]["restarts"]["vessel-3"] = 0
            else:
                before["checkpoint_measurements"] = [{"kind": "checkpoint-saved"}]
            with self.subTest(invalid=invalid), self.assertRaises(BenchmarkError):
                verify_continuation(before, after)


if __name__ == "__main__":
    unittest.main()
