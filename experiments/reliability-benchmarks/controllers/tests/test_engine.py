"""Exercise shared infrastructure through a minimal image-only application adapter."""

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import Mock, patch

from et_experiments.benchmarking.engine import BenchmarkEngine
from et_experiments.benchmarking.faults import memory_pressure_manifest, network_fault_arguments
from et_experiments.contracts import WorkloadConfig
from et_experiments.tracking.mlflow import TrackingConfig
from et_experiments.utils import BenchmarkError


class ImageOnlyAdapter:
    def __init__(self, namespace="cat-test"):
        self.config = WorkloadConfig("image-test", namespace, "trainer", ("camera-1",))
        self.tracking = None
        self.metric_aliases = {"accuracy": "cat_accuracy", "loss": "cat_loss", "participation": "contributors"}
        self.prepared = self.restored = False

    def recovery_config(self):
        return {
            "context": self.config.context,
            "namespace": self.config.namespace,
            "workloads": list(self.config.workloads),
            "progress_source": "trainer",
        }

    def prepare_scenario(self, scenario):
        return scenario

    def expected_units(self, scenario):
        return int(scenario["batches"])

    def enrich_report(self, report):
        report["application_metrics"] = {"cat_accuracy": report["workload"]["final_metrics"]["cat_accuracy"]}
        return {"model_accuracy": report["application_metrics"]["cat_accuracy"]}

    def tracking_recovery(self, state, _expected_units, _fault, _verified_at=None):
        values = (state or {}).get("history", {}).get("cat_loss", [])
        verified = any(item["timestamp"] >= 100000 for item in values)
        return {
            "status": "verified" if verified else "pending",
            "verified_at": 101 if verified else None,
            "recovery_actor": "core-mlflow-bridge",
            "recovery_seconds": 1 if verified else None,
        }

    def observe(self, _logs, _tracking_state, expected_units, _started):
        metrics = {"cat_accuracy": 0.75, "cat_loss": 0.4, "contributors": 1}
        return {
            "completed_units": 1,
            "expected_units": expected_units,
            "failed": False,
            "units": [{"batch": 0, "metrics": metrics, "duration_seconds": 1}],
            "unit_duration_seconds": {"mean": 1, "p50": 1, "p95": 1, "max": 1},
            "final_metrics": metrics,
        }

    def configure_run(self, _backend, _faults, _job_id, _scenario, _started):
        self.prepared = True

    def restore(self, _backend):
        self.restored = True

    def application_fault(self, fault):
        if fault["type"] == "image_noise":
            return {"status": "configured", "scope": "images"}
        return None


class EngineTests(unittest.TestCase):
    def run_post_release_profile(self, counters):
        app = ImageOnlyAdapter()
        base = app.observe({}, None, 2, None)
        app.observe = Mock(side_effect=[{**base, "completed_units": value} for value in counters])
        backend = Mock()
        backend.kubectl.return_value = CompletedProcess([], 0, stdout=app.config.context)
        backend.primary_logs.return_value = "progress"
        backend.collect_logs.return_value = {}
        backend.warning_events.return_value = []
        engine = BenchmarkEngine(app, backend=backend)
        engine.sampler_factory = Mock(return_value=Mock(samples=[], metrics_available=False))
        engine.faults.inject_fault = Mock(return_value={"type": "client_memory_saturation", "status": "restored"})
        engine.faults.remove_network_fault_sidecar = Mock()
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.redirect_stdout(io.StringIO()),
            patch("et_experiments.benchmarking.engine.time.sleep"),
        ):
            return engine.execute(
                "pressure",
                {
                    "batches": 2,
                    "recovery_enabled": False,
                    "require_post_fault_progress": True,
                    "fault": {"type": "client_memory_saturation", "target": "camera-1"},
                },
                Path(directory),
                2,
                False,
            )

    def test_pressure_requires_useful_progress_after_release(self):
        report = self.run_post_release_profile([0, 1, 2, 2])
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["post_fault_progress"]["status"], "verified")
        self.assertEqual(report["post_fault_progress"]["completed_at_release"], 1)
        self.assertIsNotNone(report["resilience"]["post_fault_progress_seconds"])

    def test_work_completed_under_pressure_cannot_be_claimed_as_post_release_recovery(self):
        report = self.run_post_release_profile([0, 2, 2])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["post_fault_progress"]["status"], "pending")
        self.assertIsNone(report["resilience"]["post_fault_progress_seconds"])

    def test_absent_core_incident_is_unmeasured_not_a_false_pass_or_long_timeout(self):
        result = self.run_missing_evidence([])
        self.assertIn("MTTR unmeasured", result["recovery"]["measurement_note"])

    def test_recovered_incident_without_checkpoint_evidence_has_bounded_wait(self):
        result = self.run_missing_evidence(
            [
                {
                    "workload": "trainer",
                    "status": "recovered",
                    "observed_failure_at": 1,
                    "recovered_at": 9,
                    "evidence": {},
                }
            ]
        )
        self.assertIn("not verified in time", result["recovery"]["measurement_note"])

    def run_missing_evidence(self, incidents):
        app = ImageOnlyAdapter()
        engine = BenchmarkEngine(app)
        engine.backend = Mock()
        engine.backend.kubectl.return_value = CompletedProcess([], 0, stdout=app.config.context)
        engine.backend.primary_logs.return_value = "training finished"
        engine.backend.collect_logs.return_value = {}
        engine.backend.warning_events.return_value = []
        engine.faults = Mock(interventions=[])
        engine.faults.inject_fault.return_value = {"type": "rollout_restart", "target": "trainer", "status": "injected"}
        engine.sampler_factory = Mock(return_value=Mock(samples=[], metrics_available=False))
        engine.runtime_factory = Mock(return_value=Mock(events={}))
        monitor = Mock()
        report = {"provider": "edge-toolkit-core", "incidents": incidents, "metrics": {}, "events": []}
        monitor.snapshot.return_value = report
        monitor.finish.return_value = report
        engine.monitor_factory = Mock(return_value=monitor)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            result = engine.execute(
                "missing-evidence",
                {
                    "batches": 1,
                    "require_recovery": True,
                    "require_checkpoint_restore": True,
                    "recovery_evidence_timeout_seconds": 0,
                    "fault": {"type": "rollout_restart", "target": "trainer"},
                    "slos": {"mttr_seconds": {"max": 180}},
                },
                Path(directory),
                2,
                False,
            )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["slos"]["mttr_seconds"]["status"], "not-measured")
        self.assertTrue(app.restored)
        return result

    def test_image_only_application_without_tracking(self):
        app = ImageOnlyAdapter()
        backend = Mock()
        backend.kubectl.return_value = CompletedProcess([], 0, stdout=app.config.context)
        backend.primary_logs.return_value = "image-only progress"
        backend.collect_logs.return_value = {}
        backend.warning_events.return_value = []
        engine = BenchmarkEngine(app, backend=backend)
        engine.sampler_factory = Mock(return_value=Mock(samples=[], metrics_available=False))
        engine.faults.remove_network_fault_sidecar = Mock()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            report = engine.execute("baseline", {"batches": 1, "recovery_enabled": False}, Path(directory), 2, False)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["sli"]["model_accuracy"], 0.75)
            self.assertEqual(report["resilience"]["completion_ratio"], 1)
            self.assertNotIn("training", report)
            self.assertIn("not configured", report["mlflow_annotation"])
            with (Path(directory) / "summary.csv").open() as source:
                self.assertEqual(next(csv.DictReader(source), {})["completed_units"], "1")
        self.assertTrue(app.prepared)
        self.assertTrue(app.restored)

    def test_application_bindings_are_independent(self):
        first = BenchmarkEngine(ImageOnlyAdapter("cats"))
        second = BenchmarkEngine(ImageOnlyAdapter("dogs"))
        self.assertEqual(first.backend.namespace, "cats")
        self.assertEqual(second.backend.namespace, "dogs")
        self.assertIsNot(first.faults.application, second.faults.application)

    def test_tracking_outage_waits_for_ingestion_after_training_finishes(self):
        app = ImageOnlyAdapter()
        app.tracking = TrackingConfig("cats", ("cat_loss",))
        backend = Mock()
        backend.kubectl.return_value = CompletedProcess([], 0, stdout=app.config.context)
        backend.primary_logs.return_value = "training finished"
        backend.collect_logs.return_value = {}
        backend.warning_events.return_value = []
        tracking = Mock()
        before = {"status": "FINISHED", "history": {"cat_loss": [{"step": 0, "timestamp": 50000, "value": 0.4}]}}
        after = {
            "status": "FINISHED",
            "run_id": "cats",
            "history": {"cat_loss": [{"step": 0, "timestamp": 101000, "value": 0.4}]},
        }
        tracking.mlflow_state.side_effect = [before, after, after]
        tracking.annotate_mlflow.return_value = "cats"
        engine = BenchmarkEngine(app, backend=backend, tracking=tracking)
        engine.sampler_factory = Mock(return_value=Mock(samples=[], metrics_available=False))
        engine.faults.remove_network_fault_sidecar = Mock()
        engine.faults.inject_fault = Mock(
            return_value={
                "type": "mlflow_outage",
                "target": "mlflow",
                "status": "restored",
                "started_at": "1970-01-01T00:01:40Z",
            }
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.redirect_stdout(io.StringIO()),
            patch("et_experiments.benchmarking.engine.time.sleep") as sleep,
        ):
            report = engine.execute(
                "tracking-outage",
                {
                    "batches": 1,
                    "recovery_mode": "disabled",
                    "require_tracking_recovery": True,
                    "fault": {"type": "mlflow_outage"},
                },
                Path(directory),
                2,
                False,
            )
            sleep.assert_called_once_with(2)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["tracking_recovery"]["status"], "verified")
            self.assertIn("tracking-ingestion", report["attribution"]["evidence_levels"])
            self.assertFalse(report["attribution"]["intervention_occurred"])

    def test_invalid_recovery_configuration_does_not_mutate_deployment(self):
        app = ImageOnlyAdapter()
        backend = Mock()
        engine = BenchmarkEngine(app, backend=backend)
        with self.assertRaises(BenchmarkError):
            engine.execute("bad-mode", {"recovery_mode": "unknown"}, Path("unused"), 2, False)
        backend.kubectl.assert_not_called()
        self.assertFalse(app.prepared)

    def test_nonpositive_work_target_is_rejected_before_deployment_changes(self):
        app = ImageOnlyAdapter()
        backend = Mock()
        with self.assertRaisesRegex(BenchmarkError, "positive integer"):
            BenchmarkEngine(app, backend=backend).execute("bad-target", {"batches": 0}, Path("unused"), 2, False)
        backend.kubectl.assert_not_called()
        self.assertFalse(app.prepared)

    def test_existing_domain_summary_header_is_not_overwritten_or_extended(self):
        from et_experiments.benchmarking.reports import write_reports

        report = {
            "job_id": "job",
            "scenario": "batches",
            "status": "passed",
            "fault": {"type": "none"},
            "wall_time_seconds": 2,
            "workload": ImageOnlyAdapter().observe({}, None, 1, None),
            "resilience": {"recovery_seconds": None, "completion_ratio": 1},
            "total_pod_restarts": 0,
            "total_pod_replacements": 0,
            "warning_events": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            old = "job_id,completed_rounds\nold,3\n"
            (output / "summary.csv").write_text(old)
            write_reports(output, report)
            self.assertEqual((output / "summary.csv").read_text(), old)
            with (output / "workload-summary.csv").open() as source:
                self.assertEqual(next(csv.DictReader(source))["completed_units"], "1")

    def test_application_fault_is_delegated_and_unknown_fault_rejected(self):
        engine = BenchmarkEngine(ImageOnlyAdapter())
        result = engine.faults.inject_fault({"type": "image_noise"}, "test-job")
        self.assertEqual(result["scope"], "images")
        with self.assertRaises(BenchmarkError):
            engine.faults.inject_fault({"type": "unknown"}, "test-job")

    def test_infrastructure_helpers_do_not_require_application_names(self):
        manifest = memory_pressure_manifest("job-12345678", 512, 5, "python:3-slim", "cats")
        self.assertEqual(manifest["metadata"]["namespace"], "cats")
        self.assertEqual(manifest["spec"]["template"]["spec"]["containers"][0]["image"], "python:3-slim")
        self.assertEqual(network_fault_arguments({"type": "network_partition"}), ("loss", "100%"))
