"""GPU injection is cooperative, scoped to a real accelerator, and verified with a matching receipt."""

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

from et_experiments.benchmarking.faults import KubernetesFaults
from et_experiments.benchmarking.gpu import GPU_FAULT_SCRIPT, inject_gpu_failure
from et_experiments.contracts import WorkloadConfig
from et_experiments.utils import BenchmarkError


class GPUFaultTests(unittest.TestCase):
    def setUp(self):
        self.backend = Mock(namespace="example")
        self.fault = {"target": "worker", "mode": "runtime_error"}

    def test_injects_only_an_allowlisted_worker_and_records_simulation_scope(self):
        self.backend.kubectl.return_value = subprocess.CompletedProcess(
            [], 0, json.dumps({"job_id": "job", "mode": "runtime_error", "device": "cuda", "status": "injected"})
        )
        result = inject_gpu_failure(self.backend, ("worker",), self.fault, "job")
        self.assertEqual(result["status"], "injected")
        self.assertFalse(result["hardware_failure"])
        args = self.backend.kubectl.call_args.args
        self.assertIn("deployment/worker", args)
        self.assertIn(GPU_FAULT_SCRIPT, args)
        self.assertEqual(args[-3:], ("job", "runtime_error", "60"))

    def test_rejects_targets_and_modes_before_executing_anything(self):
        for fault in ({"target": "foreign"}, {"target": "worker", "mode": "reset-driver"}):
            with self.assertRaises(BenchmarkError):
                inject_gpu_failure(self.backend, ("worker",), fault, "job")
        self.backend.kubectl.assert_not_called()

    def test_rejects_cpu_foreign_and_malformed_receipts(self):
        for result in (
            "invalid json",
            json.dumps({"job_id": "other", "mode": "runtime_error", "status": "injected", "device": "cuda"}),
            json.dumps({"job_id": "job", "mode": "runtime_error", "status": "injected", "device": "cpu"}),
        ):
            self.backend.kubectl.return_value = subprocess.CompletedProcess([], 0, result)
            with self.assertRaises(BenchmarkError):
                inject_gpu_failure(self.backend, ("worker",), self.fault, "job")

    def test_unconsumed_request_is_not_reported_as_injected(self):
        self.backend.kubectl.return_value = subprocess.CompletedProcess(
            [], 0, json.dumps({"job_id": "job", "mode": "runtime_error", "status": "not-observed"})
        )
        result = inject_gpu_failure(self.backend, ("worker",), self.fault, "job")
        self.assertEqual(result["status"], "not-observed")

    def test_generic_fault_dispatch_does_not_require_an_fl_adapter(self):
        application = Mock(config=WorkloadConfig("ctx", "example", "coordinator", ("worker",)))
        self.backend.kubectl.return_value = subprocess.CompletedProcess(
            [], 0, json.dumps({"job_id": "job", "mode": "runtime_error", "device": "cuda", "status": "injected"})
        )
        result = KubernetesFaults(self.backend, application).inject_fault({**self.fault, "type": "gpu_failure"}, "job")
        self.assertEqual(result["type"], "gpu_failure")
        self.assertEqual(result["status"], "injected")
        application.application_fault.assert_not_called()

    def test_protocol_refuses_cpu_before_writing_a_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "state.json").write_text(json.dumps({"schema_version": 1, "job_id": "job", "device": "cpu"}))
            script = GPU_FAULT_SCRIPT.replace("/tmp/et-benchmark-gpu", directory)
            result = subprocess.run(
                [sys.executable, "-c", script, "job", "runtime_error", "0"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("CPU simulation is refused", result.stderr)
            self.assertFalse((root / "request.json").exists())

    def test_protocol_timeout_cleans_up_its_unconsumed_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "state.json").write_text(json.dumps({"schema_version": 1, "job_id": "job", "device": "cuda"}))
            script = GPU_FAULT_SCRIPT.replace("/tmp/et-benchmark-gpu", directory)
            result = subprocess.run(
                [sys.executable, "-c", script, "job", "runtime_error", "0"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            self.assertEqual(json.loads(result.stdout)["status"], "not-observed")
            self.assertFalse((root / "request.json").exists())

    def test_protocol_requires_the_receipt_for_this_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "state.json").write_text(json.dumps({"schema_version": 1, "job_id": "job", "device": "mps"}))
            (root / "receipt.json").write_text(json.dumps({"request_id": "old", "status": "injected"}))
            script = GPU_FAULT_SCRIPT.replace("/tmp/et-benchmark-gpu", directory)
            with subprocess.Popen(
                [sys.executable, "-c", script, "job", "runtime_error", "3"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ) as child:
                deadline = time.monotonic() + 2
                while not (root / "request.json").exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                request = json.loads((root / "request.json").read_text())
                self.assertIsNone(child.poll())
                (root / "request.json").unlink()
                pending = root / "receipt.tmp"
                pending.write_text(json.dumps({**request, "device": "mps", "status": "injected"}))
                pending.replace(root / "receipt.json")
                stdout, stderr = child.communicate(timeout=5)
                self.assertEqual(child.returncode, 0, stderr)
                self.assertEqual(json.loads(stdout)["request_id"], request["request_id"])
