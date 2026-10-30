"""Reproducible timing, reference environments and bounded multi-client experiments."""

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import Mock, patch

from et_experiments.benchmarking.design import FaultTrigger, validate_design
from et_experiments.benchmarking.faults import KubernetesFaults
from et_experiments.benchmarking.observations import recovery_outcome
from et_experiments.benchmarking.reference import capture_reference, compare_baseline
from et_experiments.benchmarking.series import baseline_scenario, execute_series
from et_experiments.contracts import WorkloadConfig
from et_experiments.utils import BenchmarkError


class DesignTests(unittest.TestCase):
    def test_scenario_timeout_accepts_default_and_positive_finite_numbers(self):
        for scenario in ({}, {"timeout_seconds": 0.25}, {"timeout_seconds": 7200}):
            with self.subTest(scenario=scenario):
                self.assertEqual(validate_design(scenario, ("worker",)), 17)

    def test_cpu_pressure_waits_for_release_before_starting_post_fault_observation(self):
        app = Mock(config=WorkloadConfig("ctx", "ns", "primary", ("one", "two")))
        backend = Mock(namespace="ns")
        result = KubernetesFaults(backend, app).inject_fault({"type": "cpu_stress", "duration_seconds": 60}, "test-job")
        self.assertEqual(backend.kubectl.call_args.args[0], "wait")
        self.assertIn("--for=condition=Complete", backend.kubectl.call_args.args)
        self.assertIn("--timeout=150s", backend.kubectl.call_args.args)
        self.assertEqual(result["status"], "restored")
        self.assertTrue(result["released_at"])

    def test_post_fault_progress_timeout_is_bounded_before_mutation(self):
        for invalid in (0, -1, True, float("nan"), float("inf"), 7201, "240"):
            with self.subTest(value=invalid), self.assertRaises(BenchmarkError):
                validate_design({"post_fault_progress_timeout_seconds": invalid}, ("worker",))
        self.assertEqual(validate_design({"post_fault_progress_timeout_seconds": 240}, ("worker",)), 17)

    def test_event_trigger_without_progress_accepts_events_without_progress(self):
        for filters in ({}, {"workload": "worker"}):
            with self.subTest(filters=filters):
                trigger = FaultTrigger({"trigger": {"event_kind": "connected", **filters}})
                self.assertFalse(trigger.ready(0, 1, [{"id": "other", "kind": "disconnected"}]))
                event = {"id": "current", "kind": "connected", "workload": "worker"}
                self.assertTrue(trigger.ready(0, 2, [event]))
                self.assertEqual(trigger.report()["evidence"]["event_id"], "current")
                self.assertIsNone(trigger.report()["evidence"]["event_progress"])

    def test_explicit_progress_requires_a_matching_progress_value(self):
        for minimum in (0, 2):
            with self.subTest(minimum=minimum):
                trigger = FaultTrigger({"trigger": {"event_kind": "training-started", "progress": minimum}})
                event = {"id": "current", "kind": "training-started"}
                self.assertFalse(trigger.ready(0, 1, [event]))
                self.assertFalse(trigger.ready(0, 2, [{**event, "progress": minimum - 1}]))
                self.assertTrue(trigger.ready(0, 3, [{**event, "progress": minimum}]))

    def test_seeded_trigger_records_actual_observation_and_delay(self):
        fault = {
            "trigger": {
                "event_kind": "training-started",
                "workload": "worker",
                "progress": 2,
                "delay_seconds": 2,
                "jitter_seconds": 1,
            }
        }
        first, second = FaultTrigger(fault, 7), FaultTrigger(fault, 7)
        self.assertEqual(first.report(), second.report())
        self.assertFalse(
            first.ready(0, 10, [{"id": "old", "kind": "training-started", "workload": "worker", "progress": 1}])
        )
        event = {"id": "current", "kind": "training-started", "workload": "worker", "progress": 2}
        self.assertFalse(first.ready(0, 11, [event]))
        self.assertTrue(first.ready(0, 15, [event]))
        self.assertEqual(first.report()["evidence"]["observed_elapsed_seconds"], 11)
        self.assertEqual(first.report()["evidence"]["event_id"], "current")

    def test_elapsed_and_work_unit_conditions_are_both_required(self):
        trigger = FaultTrigger({"trigger": {"after_seconds": 20, "after_work_unit": 2}})
        self.assertFalse(trigger.ready(2, 21))
        self.assertFalse(trigger.ready(3, 19))
        self.assertTrue(trigger.ready(3, 20))

    def test_invalid_controls_are_rejected(self):
        for trigger in (
            {"delay_seconds": float("nan")},
            {"jitter_seconds": 61},
            {"choices": []},
            {"progress": -1},
            {"after_work_unit": "2"},
            {"event_kind": ""},
        ):
            with self.subTest(trigger=trigger), self.assertRaises(BenchmarkError):
                FaultTrigger({"trigger": trigger})
        with self.assertRaises(BenchmarkError):
            validate_design({"seed": -1}, ("one", "two"))

    def test_combined_faults_require_distinct_allowlisted_workers(self):
        faults = [{"type": "temporary_scale_down", "target": "one"}, {"type": "network_loss", "target": "two"}]
        self.assertEqual(validate_design({"fault": {"type": "combined", "faults": faults}}, ("one", "two")), 17)
        for changed in ({"target": "one"}, {"target": "foreign"}, {"type": "combined"}, {"duration_seconds": 301}):
            with self.subTest(changed=changed), self.assertRaises(BenchmarkError):
                validate_design(
                    {"fault": {"type": "combined", "faults": [faults[0], {**faults[1], **changed}]}}, ("one", "two")
                )

    def test_scale_down_restores_client_even_when_wait_is_interrupted(self):
        app = Mock(config=WorkloadConfig("ctx", "ns", "primary", ("one", "two")))
        backend = Mock(namespace="ns")
        with (
            patch("et_experiments.benchmarking.faults.time.sleep", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            KubernetesFaults(backend, app).inject_fault({"type": "temporary_scale_down", "target": "one"}, "job")
        self.assertEqual(backend.kubectl.call_args.args[-1], "--replicas=1")

    def test_standalone_scale_down_bounds_duration_and_restores_client(self):
        for options, expected in (({}, 15), ({"duration_seconds": -5}, 1), ({"duration_seconds": 1000}, 300)):
            with self.subTest(options=options):
                app = Mock(config=WorkloadConfig("ctx", "ns", "primary", ("one", "two")))
                backend = Mock(namespace="ns")
                with patch("et_experiments.benchmarking.faults.time.sleep") as sleep:
                    result = KubernetesFaults(backend, app).inject_fault(
                        {"type": "temporary_scale_down", "target": "one", **options}, "job"
                    )
                sleep.assert_called_once_with(expected)
                self.assertEqual([c.args[-1] for c in backend.kubectl.call_args_list], ["--replicas=0", "--replicas=1"])
                self.assertEqual(result["status"], "restored")

    def test_invalid_scale_down_duration_is_rejected_before_mutation(self):
        app = Mock(config=WorkloadConfig("ctx", "ns", "primary", ("one", "two")))
        backend = Mock(namespace="ns")
        with self.assertRaises(BenchmarkError):
            KubernetesFaults(backend, app).inject_fault(
                {"type": "temporary_scale_down", "target": "one", "duration_seconds": "invalid"}, "job"
            )
        backend.kubectl.assert_not_called()

    def test_combined_recovery_needs_both_targets(self):
        report = {"incidents": [{"workload": "one", "status": "recovered"}]}
        fault = {"type": "combined", "faults": [{"target": "one"}, {"target": "two"}]}
        self.assertEqual(recovery_outcome(report, {}, fault), "pending")
        report["incidents"].append({"workload": "two", "status": "recovered"})
        self.assertEqual(recovery_outcome(report, {}, fault), "recovered")

    def test_combined_children_overlap_and_preserve_both_results(self):
        app = Mock(config=WorkloadConfig("ctx", "ns", "primary", ("one", "two")))
        injector = KubernetesFaults(Mock(namespace="ns"), app)
        inject = injector.inject_fault
        barrier = threading.Barrier(2, timeout=3)

        def child_or_combined(fault, job):
            if fault["type"] == "combined":
                return inject(fault, job)
            barrier.wait()
            return {"type": fault["type"], "target": fault["target"], "status": "restored"}

        injector.inject_fault = child_or_combined
        result = injector.inject_fault(
            {
                "type": "combined",
                "faults": [
                    {"type": "temporary_scale_down", "target": "one"},
                    {"type": "network_loss", "target": "two"},
                ],
            },
            "job",
        )
        self.assertEqual(result["status"], "restored")
        self.assertEqual({r["target"] for r in result["faults"]}, {"one", "two"})


class ReferenceTests(unittest.TestCase):
    def test_snapshot_records_resources_and_hardware_without_environment_secrets(self):
        app = Mock(config=WorkloadConfig("ctx", "ns", "primary", ()))
        app.reference_conditions.return_value = {"dataset_version": "v1"}
        deployment = {
            "metadata": {"name": "primary"},
            "spec": {
                "replicas": 1,
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "primary",
                                "image": "trainer:local",
                                "env": [{"name": "TOKEN", "value": "secret"}],
                                "resources": {"limits": {"memory": "1Gi"}},
                            }
                        ]
                    }
                },
            },
        }
        pod = {
            "metadata": {"labels": {"app": "primary"}},
            "spec": {"nodeName": "node-a"},
            "status": {"containerStatuses": [{"name": "primary", "imageID": "sha256:abc"}]},
        }
        node = {
            "metadata": {"name": "node-a"},
            "status": {"allocatable": {"cpu": "4"}, "nodeInfo": {"architecture": "arm64"}},
        }
        backend = Mock()
        backend.kubectl.side_effect = [
            CompletedProcess([], 0, json.dumps({"items": [item]})) for item in (deployment, pod, node)
        ]
        reference = capture_reference(app, {"reference": {"id": "reference-v1"}}, backend)
        self.assertEqual(reference["missing"], [])
        self.assertIn("sha256:abc", json.dumps(reference))
        self.assertNotIn("secret", json.dumps(reference))
        self.assertNotIn("TOKEN", json.dumps(reference))
        self.assertEqual(reference["conditions"]["hardware"]["nodes"]["node-a"]["allocatable"]["cpu"], "4")

    def test_missing_observations_remain_explicit(self):
        app = Mock(config=WorkloadConfig("ctx", "ns", "primary", ()))
        app.reference_conditions.return_value = {}
        backend = Mock()
        backend.kubectl.return_value = CompletedProcess([], 1, "")
        self.assertEqual(
            set(capture_reference(app, {}, backend)["missing"]),
            {"declared-workload-conditions", "deployment-resources", "node-hardware-or-placement"},
        )

    def test_baseline_comparison_rejects_mismatched_or_failed_baselines(self):
        report = {"reference": {"fingerprint": "same", "missing": []}, "sli": {"throughput": 8}}
        baseline = {**report, "status": "passed", "fault": {"type": "none"}, "sli": {"throughput": 10}}
        self.assertEqual(compare_baseline(report, baseline)["metrics"]["throughput"]["ratio"], 0.8)
        for changed in (
            {"status": "failed"},
            {"fault": {"type": "delete_pod"}},
            {"reference": {"fingerprint": "other"}},
            {"reference": {"fingerprint": "same", "missing": ["hardware"]}},
        ):
            self.assertEqual(compare_baseline(report, {**baseline, **changed})["status"], "incompatible")


class SeriesTests(unittest.TestCase):
    def test_stop_on_failed_baseline_preserves_evidence_without_starting_fault(self):
        engine = Mock()
        engine.execute.return_value = {"job_id": "baseline", "status": "failed", "fault": {"type": "none"}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            public = output / "public.json"
            with patch.dict(os.environ, {"BENCHMARK_RESULT_FILE": str(public)}):
                result = execute_series(engine, "restart", {}, output, repetitions=3, paired=True, stop_on_failure=True)
            saved = json.loads((output / f"{result['series_id']}.json").read_text())
            self.assertEqual(saved["runs"], [])
            self.assertEqual(saved["summary"]["failed_baseline_runs"], 1)
            self.assertEqual(json.loads(public.read_text())["job_id"], "baseline")
        self.assertEqual(result["status"], "failed")
        engine.execute.assert_called_once()
        self.assertEqual(engine.execute.call_args.args[0], "restart-baseline")

    def test_stop_on_failed_fault_skips_later_baselines_and_repetitions(self):
        engine = Mock()
        engine.execute.side_effect = [
            {"job_id": "baseline", "status": "passed", "fault": {"type": "none"}},
            {"job_id": "fault", "status": "failed", "fault": {"type": "delete_pod"}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = execute_series(
                engine, "restart", {}, Path(directory), repetitions=3, paired=True, stop_on_failure=True
            )
        self.assertEqual(engine.execute.call_count, 2)
        self.assertEqual(len(result["runs"]), 1)
        self.assertEqual(result["status"], "failed")

    def test_failed_baseline_fails_pair_even_when_fault_run_passes(self):
        engine = Mock()
        engine.execute.side_effect = [
            {"job_id": "baseline", "status": "failed", "fault": {"type": "none"}},
            {"job_id": "fault", "status": "passed", "fault": {"type": "delete_pod"}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = execute_series(engine, "restart", {"fault": {"type": "delete_pod"}}, Path(directory), paired=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["summary"]["failed_baseline_runs"], 1)
        self.assertEqual(result["summary"]["passed_runs"], 1)

    def test_failed_cleanup_stops_later_repetitions(self):
        engine = Mock()

        def execute(name, _scenario, output, *_args):
            (output / "job.json").write_text(
                json.dumps({"job_id": "job", "scenario": name, "status": "failed", "cleanup": {"status": "failed"}})
            )
            raise BenchmarkError("cleanup failed")

        engine.execute.side_effect = execute
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(BenchmarkError, "failed cleanup"):
            execute_series(engine, "restart", {}, Path(directory), repetitions=2)
        engine.execute.assert_called_once()

    def test_baseline_preserves_workload_and_clears_fault_specific_gates(self):
        scenario = {
            "rounds": 12,
            "minimum_clients": 2,
            "fault": {"type": "network_loss", "target": "one"},
            "require_recovery": True,
            "require_participant_recovery": True,
            "slos": {"mttr": {"max": 3}},
        }
        baseline = baseline_scenario(scenario)
        self.assertEqual(baseline["rounds"], 12)
        self.assertEqual(baseline["minimum_clients"], 2)
        self.assertEqual(baseline["preparation_fault"], scenario["fault"])
        self.assertFalse(baseline["require_recovery"])
        self.assertEqual(scenario["fault"]["type"], "network_loss")

    def test_series_pairs_baselines_and_increments_timing_seed(self):
        engine = Mock()
        calls = []

        def execute(name, scenario, *_args):
            calls.append(scenario)
            return {
                "job_id": f"job-{len(calls)}",
                "scenario": name,
                "status": "passed",
                "fault": scenario["fault"],
                "reference": {"fingerprint": "same", "missing": []},
                "sli": {"latency": 2},
                "recovery": {"incidents": []},
            }

        engine.execute.side_effect = execute
        with tempfile.TemporaryDirectory() as directory:
            result = execute_series(
                engine,
                "restart",
                {"fault": {"type": "delete_pod"}, "seed": 4},
                Path(directory),
                repetitions=2,
                paired=True,
            )
            self.assertEqual([s["seed"] for s in calls], [4, 4, 5, 5])
            self.assertEqual(len(result["baselines"]), 2)
            self.assertEqual(result["runs"][0]["baseline_comparison"]["status"], "compared")
            self.assertTrue((Path(directory) / f"{result['series_id']}.json").is_file())

    def test_failed_repetition_is_retained_and_later_repetitions_continue(self):
        engine = Mock()

        def execute(name, scenario, output, *_args):
            report = {
                "job_id": f"job-{scenario['seed']}",
                "scenario": name,
                "status": "failed" if scenario["seed"] == 17 else "passed",
                "fault": scenario["fault"],
            }
            if report["status"] == "failed":
                (output / f"{report['job_id']}.json").write_text(json.dumps(report))
                raise BenchmarkError("timed out")
            return report

        engine.execute.side_effect = execute
        with tempfile.TemporaryDirectory() as directory:
            result = execute_series(
                engine, "restart", {"fault": {"type": "delete_pod"}}, Path(directory), repetitions=2
            )
        self.assertEqual(result["summary"]["failed_runs"], 1)
        self.assertEqual(result["summary"]["passed_runs"], 1)

    def test_cancellation_stops_series(self):
        engine = Mock()
        engine.execute.side_effect = KeyboardInterrupt
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(KeyboardInterrupt):
            execute_series(engine, "cancel", {}, Path(directory), repetitions=3)
        engine.execute.assert_called_once()

    def test_failed_report_recovery_preserves_detailed_evidence(self):
        engine = Mock()

        def execute(name, _scenario, output, *_args):
            public = {"job_id": "bench-failed", "scenario": name, "status": "failed"}
            Path(os.environ["BENCHMARK_RESULT_FILE"]).write_text(json.dumps({**public, "job_id": "a"}))
            (output / "b-decoy.json").write_text(json.dumps(public))
            detailed = {**public, "workload": {"completed_units": 3}, "samples": [{"ready": False}], "events": [1]}
            (output / "bench-failed.json").write_text(json.dumps(detailed))
            raise BenchmarkError("workload failed")

        engine.execute.side_effect = execute
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.dict(os.environ, {"BENCHMARK_RESULT_FILE": str(output / "a.json")}):
                result = execute_series(engine, "restart", {}, output)
            persisted = json.loads((output / "bench-failed.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(persisted["workload"], {"completed_units": 3})
        self.assertEqual(persisted["samples"], [{"ready": False}])
        self.assertEqual(persisted["events"], [1])
