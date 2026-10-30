"""Verify benchmark integration with Core recovery decisions and evidence."""

import datetime as dt
import json
import unittest
from unittest.mock import Mock

from et_experiments.backends.core_recovery import CoreRecoveryMonitor, CoreRecoveryRuntime, normalize_report
from et_experiments.benchmarking.observations import recovery_outcome
from et_experiments.recovery.core import checkpoint_metrics, evaluate_slos, summarize_runs
from et_experiments.utils import BenchmarkError
from recovery_fixtures import core_report


class RecoveryReportTests(unittest.TestCase):
    def test_restoration_can_roll_back_unsaved_progress(self):
        raw = core_report()
        raw["incidents"][0].update(onset=438, detected=442, recovered=540, baseline=4)
        for item in raw["checkpoint_measurements"]:
            item.update(observed_at=471, progress=3)
        raw["checkpoint_measurements"].insert(
            0,
            {
                **raw["checkpoint_measurements"][0],
                "kind": "checkpoint-saved",
                "instance": "original",
                "observed_at": 328,
            },
        )
        report = normalize_report(raw, "job", "observe-only")
        self.assertEqual(
            recovery_outcome(report, {"require_checkpoint_restore": True}, {"target": "worker"}), "recovered"
        )
        evidence = report["incidents"][0]["evidence"]
        self.assertEqual(evidence["checkpoint_restored_at"], 471)
        self.assertEqual(evidence["resume_completed_at"], 471)
        self.assertEqual(report["core_report"], raw)

    def test_delayed_checkpoint_pair_is_accepted_without_changing_core_mttr(self):
        raw = core_report()
        for item in raw["checkpoint_measurements"]:
            item["observed_at"] = 77
        report = normalize_report(raw, "job", "observe-only")
        self.assertEqual(
            recovery_outcome(report, {"require_checkpoint_restore": True}, {"target": "worker"}), "recovered"
        )
        self.assertEqual(report["metrics"]["mttr_seconds"], 8)

    def test_mismatched_old_or_next_incident_checkpoint_pairs_are_not_credited(self):
        for case in ("different-process", "old-process", "next-incident", "before-incident", "other-workload"):
            with self.subTest(case=case):
                raw = core_report()
                for item in raw["checkpoint_measurements"]:
                    item["observed_at"] = 77
                if case == "different-process":
                    raw["checkpoint_measurements"][1]["instance"] = "other"
                elif case == "old-process":
                    raw["checkpoint_measurements"].append(
                        {**raw["checkpoint_measurements"][0], "kind": "checkpoint-saved", "observed_at": 0}
                    )
                elif case == "next-incident":
                    raw["incidents"].append({**raw["incidents"][0], "onset": 70})
                elif case == "before-incident":
                    for item in raw["checkpoint_measurements"]:
                        item["observed_at"] = 0
                else:
                    for item in raw["checkpoint_measurements"]:
                        item["workload"] = "other"
                report = normalize_report(raw, "job", "observe-only")
                self.assertNotIn("resume_completed_at", report["incidents"][0]["evidence"])

    def test_core_recovery_and_observer_clock_are_preserved(self):
        report = normalize_report(core_report(), "job", "observe-only")
        self.assertEqual(report["metrics"]["mttr_seconds"], 8)
        self.assertEqual(report["metrics"]["failure_detection_mean_seconds"], 2)
        self.assertEqual(report["incidents"][0]["observed_failure_at"], 1)
        self.assertEqual(report["clock_basis"], "observer-monotonic-seconds")
        self.assertEqual(report["metrics"]["observation_errors"], 2)
        self.assertIsNone(report["metrics"]["observed_availability_ratio"])

    def test_readiness_and_old_completion_cannot_invent_recovery(self):
        for status in ("suspected", "transient", "detected", "unrecovered"):
            with self.subTest(status=status):
                report = normalize_report(core_report(status=status), "job", "observe-only")
                self.assertEqual(report["incidents"][0]["status"], status)
                self.assertIsNone(report["metrics"]["mttr_seconds"])
                self.assertIsNone(report["incidents"][0]["ready_at"])

    def test_wrong_job_mode_schema_or_clock_is_rejected(self):
        for key, value in (("job_id", "old"), ("schema_version", 99), ("mode", "disabled"), ("clock_basis", "wall")):
            with self.subTest(key=key), self.assertRaises(BenchmarkError):
                normalize_report({**core_report(), key: value}, "job", "observe-only")

    def test_core_checkpoint_timers_and_fault_gate_use_same_clock(self):
        report = normalize_report(core_report(), "job", "observe-only")
        self.assertEqual(checkpoint_metrics(report["events"])["checkpoint_load_mean_seconds"], 0.125)
        scenario = {"require_checkpoint_restore": True}
        fault = {"target": "worker", "started_at": "2026-10-19T00:00:00Z"}
        self.assertEqual(recovery_outcome(report, scenario, fault), "recovered")
        report["incidents"][0]["evidence"].pop("resume_completed_at")
        self.assertEqual(recovery_outcome(report, scenario, fault), "pending")

    def test_pre_fault_incidents_do_not_satisfy_recovery(self):
        report = normalize_report(core_report(), "job", "observe-only")
        self.assertEqual(recovery_outcome(report, {}, {"target": "worker", "core_incident_baseline": 1}), "pending")

    def test_transient_fault_is_unmeasured_not_a_recovery_timeout(self):
        report = normalize_report(core_report(status="transient"), "job", "observe-only")
        self.assertEqual(recovery_outcome(report, {}, {"target": "worker"}), "not-detected")
        self.assertIsNone(report["metrics"]["mttr_seconds"])

    def test_missing_sli_does_not_pass_slo(self):
        result = evaluate_slos({}, {"model_accuracy": {"min": 0.8}})
        self.assertEqual(result["model_accuracy"]["status"], "not-measured")

    def test_transients_and_legacy_observer_results_do_not_distort_core_mttr(self):
        report = normalize_report(core_report(), "job", "observe-only")
        report["incidents"].append({"status": "transient"})
        result = {"status": "passed", "recovery": report}
        summary = summarize_runs([result])
        self.assertEqual(summary["incidents"], 1)
        legacy = {"status": "passed", "recovery": {"provider": "research-controller", "incidents": []}}
        summary = summarize_runs([result, legacy])
        self.assertTrue(summary["mixed_recovery_providers"])
        self.assertIsNone(summary["mttr_seconds"])

    def test_mttr_pools_incidents_and_preserves_unrecovered_runs(self):
        results = [
            {"status": "passed", "recovery": normalize_report(core_report(), "job", "observe-only")},
            {
                "status": "failed",
                "recovery": normalize_report(core_report(status="unrecovered"), "job", "observe-only"),
            },
        ]
        summary = summarize_runs(results)
        self.assertEqual(summary["mttr_seconds"], 8)
        self.assertEqual(summary["unrecovered_incidents"], 1)
        self.assertEqual(summary["recovery_success_ratio"], 0.5)


class CoreRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.backend = Mock(namespace="demo")
        self.backend.kubectl.return_value.returncode = 0
        self.backend.collect_logs.return_value = {}
        self.config = {
            "supervisor_deployment": "recovery-supervisor",
            "workloads": ["worker"],
            "started_at": dt.datetime.now(dt.timezone.utc),
            "job_id": "manual-job",
        }
        self.runtime = CoreRecoveryRuntime(self.backend, self.config, "job", "observe-only")
        self.pod = {"metadata": {"uid": "one"}, "status": {"containerStatuses": [{"restartCount": 0}]}}
        self.role = {
            "rules": [
                {
                    "apiGroups": ["apps"],
                    "resources": ["deployments"],
                    "resourceNames": ["worker"],
                    "verbs": ["get"],
                }
            ]
        }

    def test_disabled_mode_explicitly_stops_deployed_supervision(self):
        self.runtime.mode = "disabled"
        self.runtime.prepare()
        self.runtime.start()
        calls = [call.args for call in self.backend.kubectl.call_args_list]
        self.assertTrue(any("--replicas=0" in call for call in calls))
        self.assertFalse(any("--replicas=1" in call for call in calls))
        self.assertFalse(any("rollout" in call for call in calls))

    def test_start_configures_only_the_supervisor_and_preserves_the_job_state_path(self):
        def command(*arguments, **_kwargs):
            if arguments[0] == "get" and arguments[1].startswith("role/"):
                return Mock(stdout=json.dumps(self.role), returncode=0)
            if arguments[:2] == ("get", "pods"):
                return Mock(stdout=json.dumps({"items": [self.pod]}), returncode=0)
            if arguments[0] == "exec":
                return Mock(stdout=json.dumps(core_report()), returncode=0)
            return Mock(stdout="", returncode=0)

        self.backend.kubectl.side_effect = command
        self.runtime.start()
        calls = [call.args for call in self.backend.kubectl.call_args_list]
        self.assertTrue(any("RECOVERY_STATE_FILE=/app/state/job.json" in call for call in calls))
        self.assertTrue(any("RECOVERY_MODE=observe-only" in call for call in calls))
        self.assertFalse(any("delete" in call or "rm" in call for call in calls))
        self.assertFalse(any("deployment/worker" in call for call in calls))
        self.assertTrue(self.runtime.active)

    def test_already_stopped_supervisor_is_safe_but_remaining_pods_are_not(self):
        self.backend.kubectl.side_effect = [Mock(), Mock(), Mock(returncode=1), Mock(stdout='{"items": []}')]
        self.runtime.prepare()
        self.backend.kubectl.side_effect = [
            Mock(),
            Mock(),
            Mock(returncode=1),
            Mock(stdout=json.dumps({"items": [self.pod]})),
        ]
        with self.assertRaisesRegex(BenchmarkError, "did not stop"):
            self.runtime.prepare()

    def test_observe_only_and_assisted_permissions_are_allowlisted(self):
        self.backend.kubectl.return_value = Mock(stdout=json.dumps(self.role))
        for mode, expected in (("observe-only", ["get"]), ("controller-assisted", ["get", "patch"])):
            self.runtime.permissions(mode)
            command = self.backend.kubectl.call_args.args
            self.assertEqual(json.loads(command[-1])[0]["value"], expected)
        self.role["rules"][0]["resourceNames"] = ["unrelated"]
        self.backend.kubectl.return_value.stdout = json.dumps(self.role)
        with self.assertRaises(BenchmarkError):
            self.runtime.permissions("controller-assisted")

    def test_core_report_is_authoritative_and_session_replacement_fails_closed(self):
        self.backend.kubectl.side_effect = [
            Mock(stdout=json.dumps({"items": [self.pod]})),
            Mock(stdout=json.dumps(core_report())),
            Mock(stdout=json.dumps({"items": [{**self.pod, "metadata": {"uid": "replacement"}}]})),
        ]
        self.assertEqual(self.runtime.snapshot()["metrics"]["mttr_seconds"], 8)
        with self.assertRaisesRegex(BenchmarkError, "restarted"):
            self.runtime.snapshot()
        self.assertFalse(any("restart" in call.args for call in self.backend.kubectl.call_args_list))

    def test_same_pod_container_restart_also_changes_the_observer_session(self):
        self.runtime.session = ("one", 0)
        self.pod["status"]["containerStatuses"][0]["restartCount"] = 1
        self.backend.kubectl.return_value.stdout = json.dumps({"items": [self.pod]})
        with self.assertRaisesRegex(BenchmarkError, "restarted"):
            self.runtime.snapshot()

    def test_malformed_or_foreign_report_is_not_treated_as_success(self):
        for report in ({}, {**core_report(), "job_id": "old"}):
            self.backend.kubectl.side_effect = [
                Mock(stdout=json.dumps({"items": [self.pod]})),
                Mock(stdout=json.dumps(report)),
            ]
            with self.assertRaises(BenchmarkError):
                self.runtime.snapshot()

    def test_missing_core_supervisor_fails_before_workload_changes(self):
        self.backend.kubectl.side_effect = BenchmarkError("not found")
        with self.assertRaisesRegex(BenchmarkError, "not found"):
            self.runtime.prepare()
        self.backend.kubectl.assert_called_once()

    def test_report_read_uses_job_specific_file_without_resetting_budget(self):
        self.runtime.read_state()
        command = self.backend.kubectl.call_args.args
        self.assertEqual(command[-2:], ("cat", "/app/state/job.json"))
        self.assertFalse(hasattr(self.runtime, "restart"))
        with self.assertRaises(BenchmarkError):
            CoreRecoveryRuntime(self.backend, self.config, "../escape", "observe-only")

    def test_finish_stops_core_even_when_report_collection_fails(self):
        self.backend.kubectl.side_effect = [BenchmarkError("report unavailable"), Mock()]
        with self.assertRaisesRegex(BenchmarkError, "report unavailable"):
            CoreRecoveryMonitor(self.runtime, 2).finish()
        self.assertIn("--replicas=0", self.backend.kubectl.call_args.args)


if __name__ == "__main__":
    unittest.main()
