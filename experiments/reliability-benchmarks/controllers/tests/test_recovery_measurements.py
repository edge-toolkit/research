"""Separate service continuity, participant contributions and checkpoint continuation evidence."""

import datetime as dt
import unittest

from et_experiments.benchmarking.evidence import processing_metrics, recovery_evidence
from et_experiments.recovery.core import summarize_runs
from et_experiments.recovery.statistics import distribution, success_interval


def event(kind, at, progress=0, instance="old", **fields):
    return {"kind": kind, "at": at, "progress": progress, "instance": instance, "workload": "primary", **fields}


def report(events):
    def timestamp(seconds):
        return dt.datetime.fromtimestamp(seconds, dt.timezone.utc).isoformat()

    return {
        "fault": {
            "type": "temporary_scale_down",
            "target": "worker",
            "started_at": timestamp(10),
            "finished_at": timestamp(20),
        },
        "diagnostic_events": events,
        "workload": {"expected_units": 12},
    }


class EvidenceTests(unittest.TestCase):
    def test_processing_interval_includes_recovery_and_excludes_deployment_setup(self):
        events = [
            event("operation-started", 10, 0),
            event("progress", 12, 1),
            event("resume-completed", 25, 1, "new"),
            event("progress", 40, 3, "new"),
        ]
        self.assertEqual(processing_metrics(events, "primary", 3)["processing_time_seconds"], 30)
        self.assertEqual(processing_metrics(events, "primary", 3)["processing_units_per_minute"], 6)
        self.assertIsNone(processing_metrics(events[1:], "primary", 3)["processing_time_seconds"])

    def test_service_continuity_and_participant_return_are_independent(self):
        evidence = recovery_evidence(report([event("progress", 15, 2)]), "primary")
        self.assertEqual(evidence["service_continuity"]["status"], "observed")
        self.assertEqual(evidence["participant_recovery"]["worker"]["status"], "not-observed")
        evidence = recovery_evidence(
            report(
                [
                    event("contribution-ready", 21, 3, "new", workload="worker"),
                    event("contribution-accepted", 22, 3, participant="worker", participant_instance="new"),
                ]
            ),
            "primary",
        )
        self.assertEqual(evidence["service_continuity"]["status"], "not-observed")
        self.assertEqual(evidence["participant_recovery"]["worker"]["status"], "verified")

    def test_only_post_release_accepted_target_contributions_verify_return(self):
        events = [
            event("contribution-accepted", 19, participant="worker"),
            event("contribution-accepted", 22, participant="other"),
        ]
        evidence = recovery_evidence(report(events), "primary")
        self.assertEqual(evidence["participant_recovery"]["worker"]["status"], "not-observed")

    def test_participant_return_requires_progress_on_both_events_including_zero(self):
        for sent_progress, accepted_progress, expected in (
            (None, None, "not-observed"),
            (None, 1, "not-observed"),
            (1, None, "not-observed"),
            (0, 0, "verified"),
        ):
            with self.subTest(sent=sent_progress, accepted=accepted_progress):
                sent = event("contribution-ready", 21, instance="new", workload="worker")
                accepted = event("contribution-accepted", 22, participant="worker", participant_instance="new")
                if sent_progress is None:
                    sent.pop("progress")
                else:
                    sent["progress"] = sent_progress
                if accepted_progress is None:
                    accepted.pop("progress")
                else:
                    accepted["progress"] = accepted_progress
                result = recovery_evidence(report([sent, accepted]), "primary")
                self.assertEqual(result["participant_recovery"]["worker"]["status"], expected)

    def test_stale_checkpoint_records_retained_progress_and_new_useful_work(self):
        events = [
            event("checkpoint-saved", 4, 3, bytes=1024),
            event("progress", 8, 4),
            event("checkpoint-restored", 21, 3, "new"),
            event("resume-completed", 22, 3, "new"),
            event("progress", 24, 4, "new"),
        ]
        checkpoint = recovery_evidence(report(events), "primary")["checkpoint"]
        self.assertEqual(checkpoint["status"], "verified-progress-metadata")
        self.assertEqual(checkpoint["checkpoint_age_at_fault_seconds"], 6)
        self.assertEqual(checkpoint["unsaved_completed_units"], 1)
        self.assertEqual(checkpoint["checkpoint_bytes"], 1024)
        self.assertTrue(checkpoint["resumed_unfinished_work"])

    def test_completed_state_and_cross_process_resume_do_not_prove_continuation(self):
        events = [
            event("checkpoint-saved", 4, 12),
            event("checkpoint-restored", 21, 12, "new"),
            event("resume-completed", 22, 12, "new"),
        ]
        self.assertFalse(recovery_evidence(report(events), "primary")["checkpoint"]["resumed_unfinished_work"])
        events[-1]["instance"] = "unrelated"
        self.assertEqual(recovery_evidence(report(events), "primary")["checkpoint"]["status"], "not-observed")

    def test_observer_relative_clocks_are_excluded_from_wall_clock_analysis(self):
        events = [event("progress", 15, 2, clock_basis="observer-monotonic-seconds")]
        evidence = recovery_evidence(report(events), "primary")
        self.assertEqual(evidence["service_continuity"]["status"], "not-observed")


class StatisticsTests(unittest.TestCase):
    def test_small_samples_have_explicit_tail_quantile_limits(self):
        self.assertIsNone(distribution([1, 2, 3])["p95"])
        self.assertIsNotNone(distribution(list(range(20)))["p95"])
        self.assertIsNone(distribution(list(range(20)))["p99"])
        self.assertEqual(distribution([float("nan"), -1, True])["samples"], 0)

    def test_success_interval_expresses_uncertainty_even_when_all_tests_pass(self):
        interval = success_interval(3, 3)
        self.assertLess(interval["lower"], 0.9)
        self.assertAlmostEqual(interval["upper"], 1)
        self.assertIsNone(success_interval(0, 0))

    def test_different_faults_or_reference_conditions_are_not_pooled(self):
        first = {
            "status": "passed",
            "fault": {"type": "delete_pod"},
            "reference": {"fingerprint": "one"},
            "recovery": {"incidents": [{"status": "recovered", "observed_failure_at": 1, "recovered_at": 9}]},
        }
        same = summarize_runs([first, first])
        self.assertEqual(same["mttr_seconds"], 8)
        self.assertEqual(same["recovery_distribution"]["samples"], 2)
        for change in ({"fault": {"type": "network_loss"}}, {"reference": {"fingerprint": "two"}}):
            summary = summarize_runs([first, {**first, **change}])
            self.assertIsNone(summary["mttr_seconds"])
            self.assertEqual(len(summary["comparable_groups"]), 2)

    def test_unresolved_and_undetected_runs_remain_in_summary(self):
        runs = [
            {"status": "failed", "recovery": {"incidents": []}},
            {"status": "failed", "recovery": {"incidents": [{"status": "unrecovered"}]}},
        ]
        summary = summarize_runs(runs)
        self.assertEqual(summary["failed_runs"], 2)
        self.assertEqual(summary["comparable_groups"][0]["runs_without_detected_incidents"], 1)
        self.assertEqual(summary["recovery_success_ratio"], 0)

    def test_baseline_deltas_keep_negative_quality_changes(self):
        result = {
            "status": "passed",
            "sli": {"accuracy": 0.6},
            "baseline_comparison": {"status": "compared", "metrics": {"accuracy": {"delta": -0.2}}},
        }
        group = summarize_runs([result])["comparable_groups"][0]
        self.assertEqual(group["matched_baseline_runs"], 1)
        self.assertEqual(group["baseline_delta_distributions"]["accuracy"]["mean"], -0.2)
