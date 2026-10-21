"""Exercise real tracking commands and fault recovery without external services."""

import json
import unittest
from subprocess import CompletedProcess
from types import SimpleNamespace
from unittest.mock import Mock, patch

from et_experiments.benchmarking.faults import KubernetesFaults
from et_experiments.contracts import WorkloadConfig
from et_experiments.tracking.mlflow import MLflowTracking, TrackingConfig, mlflow_benchmark_payload


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.backend = Mock(namespace="test")
        self.backend.kubectl.return_value = CompletedProcess([], 0, stdout=json.dumps({"run_id": "test-run"}))
        self.config = TrackingConfig("test", ("loss",), deployment="tracking", bridge_deployment="bridge")
        self.tracking = MLflowTracking(self.backend, self.config)

    def test_tracking_read_executes_a_valid_script_via_backend(self):
        self.assertEqual(self.tracking.mlflow_state("job"), {"run_id": "test-run"})
        args = self.backend.kubectl.call_args.args
        self.assertIn("deployment/tracking", args)
        script = args[args.index("-c") + 1]
        compile(script, "<tracking-read>", "exec")
        self.assertEqual(json.loads(args[-1]), ["loss"])

    def test_tracking_annotation_executes_a_valid_script(self):
        self.backend.kubectl.return_value = CompletedProcess([], 0, stdout="test-run")
        report = {
            "mlflow_run_id": "test-run",
            "attribution": {
                "recovery_mode": "observe-only",
                "scope": "combined-deployed-system",
                "recovery_actors": ["kubernetes"],
                "evidence_levels": ["application-progress"],
                "intervention_occurred": False,
            },
        }
        self.assertEqual(self.tracking.annotate_mlflow("restart", report), "test-run")
        args = self.backend.kubectl.call_args.args
        compile(args[args.index("-c") + 1], "<tracking-annotation>", "exec")
        payload = json.loads(args[-1])
        self.assertEqual(payload["tags"]["benchmark.recovery_mode"], "observe-only")
        self.assertEqual(payload["tags"]["benchmark.recovery_actors"], "kubernetes")

    def test_mlflow_outage_only_restarts_bridge_when_explicit(self):
        application = SimpleNamespace(
            config=WorkloadConfig("ctx", "test", "trainer", (), ("tracking", "bridge")), tracking=self.config
        )
        for assisted in (False, True):
            with self.subTest(assisted=assisted):
                backend = Mock(namespace="test")
                faults = KubernetesFaults(backend, application)
                with patch("et_experiments.benchmarking.faults.time.sleep"):
                    result = faults.inject_fault({"type": "mlflow_outage", "restart_bridge": assisted}, "job")
                restart = [call for call in backend.kubectl.call_args_list if call.args[:2] == ("rollout", "restart")]
                self.assertEqual(len(restart), int(assisted))
                self.assertEqual(len(result["interventions"]), int(assisted))

    def test_tracking_recovery_metrics_and_tags(self):
        payload = mlflow_benchmark_payload(
            {
                "tracking_recovery": {
                    "status": "verified",
                    "recovery_seconds": 15,
                    "forced_bridge_restart": False,
                    "recovery_actor": "core-mlflow-bridge",
                }
            },
            "outage",
        )
        self.assertEqual(payload["metrics"]["benchmark.tracking_recovery_seconds"], 15)
        self.assertEqual(payload["metrics"]["benchmark.tracking_recovery_verified"], 1)
        self.assertEqual(payload["tags"]["benchmark.forced_bridge_restart"], "False")
