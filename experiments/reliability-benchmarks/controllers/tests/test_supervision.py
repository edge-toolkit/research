"""New supervisor evidence remains separate from MTTR and survives only verified context continuation."""

import copy
import datetime as dt
import json
import unittest
from unittest.mock import Mock

from et_experiments.backends.core_recovery import CoreRecoveryRuntime, normalize_report
from et_experiments.benchmarking.observations import stability_outcome
from et_experiments.benchmarking.series import baseline_scenario
from et_experiments.recovery.supervision import verify_continuation
from et_experiments.utils import BenchmarkError
from recovery_fixtures import core_report


def durable_report():
    raw = core_report()
    raw["incidents"][0]["stability"] = {"status": "stable", "verified_at": 19}
    raw["context"] = {
        "schema_version": 1,
        "configuration": {"job_id": "job", "resilience": {"stability_seconds": 10}},
        "jitter_seed": 123,
        "restored_sessions": 0,
        "elapsed_seconds": 19,
    }
    raw["assessments"] = {"worker": {"classification": "healthy", "reason": "healthy"}}
    raw["observation_gaps"] = []
    return raw


def continued_report(previous):
    raw = copy.deepcopy(previous)
    raw["clock_basis"] = "observer-elapsed-seconds-with-restart-gaps"
    raw["context"].update(restored_sessions=1, elapsed_seconds=25)
    raw["observation_gaps"] = [{"started_at": 19, "ended_at": 25, "reason": "supervisor restart"}]
    return raw


class SupervisionTests(unittest.TestCase):
    def test_manual_context_restoration_keeps_declared_dependency_permissions(self):
        backend = Mock(namespace="demo")
        rule = {"apiGroups": ["apps"], "resources": ["deployments"], "resourceNames": ["worker"], "verbs": ["get"]}
        dependency = {**rule, "resourceNames": ["hub"]}
        backend.kubectl.return_value.stdout = json.dumps({"rules": [rule, dependency]})
        manual = {"dependencies": {}, "backoff_multiplier": 1}
        deployed = {"dependencies": {"worker": ["hub"]}, "backoff_multiplier": 2}
        config = {
            "workloads": ["worker"],
            "supervisor_deployment": "recovery-supervisor",
            "job_id": "manual",
            "mode": "observe-only",
            "deployment_resilience": deployed,
            "default_resilience": manual,
            "resilience": {**deployed, "stability_seconds": 10},
        }
        runtime = CoreRecoveryRuntime(backend, config, "benchmark", "observe-only")
        runtime.restore_defaults()
        calls = [call.args for call in backend.kubectl.call_args_list]
        environment = next(args for args in calls if args[:2] == ("set", "env"))
        restored = next(value.split("=", 1)[1] for value in environment if value.startswith("RECOVERY_RESILIENCE="))
        self.assertEqual(json.loads(restored), manual)
        self.assertIn("RECOVERY_JOB_ID=manual", environment)
        self.assertIn("RECOVERY_STATE_FILE=/app/state/recovery.json", environment)
        self.assertEqual(dependency["verbs"], ["get"])
        self.assertEqual(
            [call.args[0] for call in backend.wait_rollouts.call_args_list],
            [("worker",), ("recovery-supervisor",)],
        )
        backend.wait_rollouts.side_effect = [None, BenchmarkError("manual context rejected")]
        with self.assertRaisesRegex(BenchmarkError, "manual context rejected"):
            runtime.restore_defaults()

    def test_supporting_stability_does_not_change_mttr_or_success_rate(self):
        for state in ("stable", "unstable", "pending", "incomplete", "disabled"):
            with self.subTest(state=state):
                raw = durable_report()
                raw["incidents"][0]["stability"]["status"] = state
                report = normalize_report(raw, "job", "observe-only")
                self.assertEqual(report["metrics"]["mttr_seconds"], 8)
                self.assertEqual(report["metrics"]["recovery_success_ratio"], 1)
                expected = None if state == "disabled" else float(state == "stable")
                self.assertEqual(report["metrics"]["recovery_stability_verified_ratio"], expected)
                self.assertEqual(report["assessments"], raw["assessments"])
        legacy = normalize_report(core_report(), "job", "observe-only")
        self.assertEqual(legacy["metrics"]["recovery_stability_unreported_incidents"], 1)
        self.assertIsNone(legacy["metrics"]["recovery_stability_verified_ratio"])

    def test_stability_gate_and_matched_baseline_do_not_invent_recovery(self):
        scenario = {"require_recovery_stability": True, "recovery_resilience": {"stability_seconds": 10}}
        for state, expected in (
            ("stable", "stable"),
            ("pending", "pending"),
            ("unstable", "unstable"),
            ("incomplete", "incomplete"),
            ("disabled", "incomplete"),
        ):
            raw = durable_report()
            raw["incidents"][0]["stability"]["status"] = state
            report = normalize_report(raw, "job", "observe-only")
            self.assertEqual(stability_outcome(report, scenario, {"target": "worker"}), expected)
        baseline = baseline_scenario(scenario)
        self.assertFalse(baseline["require_recovery_stability"])
        self.assertEqual(baseline["recovery_resilience"], scenario["recovery_resilience"])

    def test_stability_before_recovery_or_unknown_status_is_rejected(self):
        for value in ({"status": "stable", "verified_at": 8}, {"status": "made-up"}, "stable"):
            raw = durable_report()
            raw["incidents"][0]["stability"] = value
            with self.assertRaises(BenchmarkError):
                normalize_report(raw, "job", "observe-only")

    def test_restored_clock_records_gap_without_resetting_recovery_times(self):
        raw = continued_report(durable_report())
        report = normalize_report(raw, "job", "observe-only")
        self.assertEqual(report["metrics"]["mttr_seconds"], 8)
        self.assertEqual(report["metrics"]["supervision_closed_gap_seconds"], 6)
        self.assertEqual(report["restored_sessions"], 1)
        raw.pop("context")
        with self.assertRaisesRegex(BenchmarkError, "durable recovery context"):
            normalize_report(raw, "job", "observe-only")

    def test_continuation_requires_preserved_identity_incidents_and_budget(self):
        previous = durable_report()
        previous["restart_budget"]["restarts"] = {"worker": 1}
        current = continued_report(previous)
        verify_continuation(previous, current)
        for field in ("seed", "configuration", "sessions", "incidents", "budget", "checkpoint"):
            invalid = copy.deepcopy(current)
            if field == "seed":
                invalid["context"]["jitter_seed"] = 999
            elif field == "configuration":
                invalid["context"]["configuration"]["job_id"] = "foreign"
            elif field == "sessions":
                invalid["context"]["restored_sessions"] = 0
            elif field == "incidents":
                invalid["incidents"] = []
            elif field == "budget":
                invalid["restart_budget"]["restarts"] = {}
            else:
                invalid["checkpoint_measurements"] = []
            with self.subTest(field=field), self.assertRaises(BenchmarkError):
                verify_continuation(previous, invalid)

    def test_runtime_accepts_verified_replacement_and_keeps_dependency_access_read_only(self):
        previous = durable_report()
        current = continued_report(previous)
        backend = Mock(namespace="demo")
        config = {"workloads": ["worker"], "started_at": dt.datetime.now(dt.timezone.utc)}
        runtime = CoreRecoveryRuntime(backend, config, "job", "observe-only")
        runtime.last_report = normalize_report(previous, "job", "observe-only")
        runtime.session = ("old", 0)
        backend.collect_logs.return_value = {}
        backend.kubectl.side_effect = [
            Mock(
                stdout=json.dumps(
                    {"items": [{"metadata": {"uid": "new"}, "status": {"containerStatuses": [{"restartCount": 0}]}}]}
                )
            ),
            Mock(stdout=json.dumps(current)),
        ]
        self.assertEqual(runtime.snapshot()["restored_sessions"], 1)
        runtime.config["resilience"] = {"dependencies": {"worker": ["hub"]}}
        rule = {"apiGroups": ["apps"], "resources": ["deployments"], "resourceNames": ["worker"], "verbs": ["get"]}
        dependency = {**rule, "resourceNames": ["hub"]}
        backend.kubectl.side_effect = None
        backend.kubectl.return_value.stdout = json.dumps({"rules": [rule, dependency]})
        runtime.permissions("controller-assisted")
        patch = json.loads(backend.kubectl.call_args.args[-1])
        self.assertEqual(patch, [{"op": "replace", "path": "/rules/0/verbs", "value": ["get", "patch"]}])
        dependency["verbs"] = ["get", "patch"]
        backend.kubectl.return_value.stdout = json.dumps({"rules": [rule, dependency]})
        with self.assertRaises(BenchmarkError):
            runtime.permissions("controller-assisted")
