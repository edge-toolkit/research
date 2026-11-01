"""Device portability and safe accelerator failure injection, without a dataset or physical GPU faults."""

import json
import os
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from harbour_fl import checkpoints, devices, learning, settings
from harbour_fl.client import ClientRole


class DeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_explicit_gpu_never_silently_becomes_cpu(self):
        """Verify explicit gpu never silently becomes cpu."""
        fake = Mock()
        fake.cuda.is_available.return_value = False
        fake.backends.mps.is_available.return_value = False
        for name in ("cuda", "mps"):
            with self.assertRaisesRegex(RuntimeError, "fallback is disabled"):
                devices.select_device(fake, name)
        fake.device.side_effect = lambda name: name
        self.assertEqual(devices.select_device(fake, "auto"), "cpu")
        fake.cuda.is_available.return_value = True
        self.assertEqual(devices.select_device(fake, "auto"), "cuda")
        fake.cuda.is_available.return_value = False
        fake.backends.mps.is_available.return_value = True
        self.assertEqual(devices.select_device(fake, "auto"), "mps")

    def test_cpu_training_evaluation_and_checkpoint_roundtrip(self):
        self.exercise_device("cpu")

    def test_available_accelerators_train_and_export_portable_updates(self):
        """Verify available accelerators train and export portable updates."""
        available = [
            name
            for name, enabled in (
                ("cuda", torch.cuda.is_available()),
                ("mps", torch.backends.mps.is_available()),
            )
            if enabled
        ]
        if not available:
            self.skipTest("No CUDA/MPS accelerator is exposed to this Python process")
        for name in available:
            with self.subTest(device=name):
                self.exercise_device(name)

    def exercise_device(self, name):
        """Train and serialize a small fusion head on one explicitly selected accelerator backend."""
        model = learning._new_model(torch, 2).to(devices.select_device(torch, name))
        batch = (torch.rand(2, 960), torch.rand(2, 960), torch.tensor([0, 1]))
        with (
            patch.dict(os.environ, {"HARBOUR_FL_ADAPTATION": "true", "HARBOUR_FL_LOCAL_EPOCHS": "1"}),
            patch.object(learning, "_batches", return_value=[[{}, {}]]),
            patch.object(learning, "_materialize", return_value=batch),
        ):
            trained = learning._train_local(model, [{"class_weights": [1.0, 1.0]}, {}], ["a", "b"], 0, torch)
            evaluated = learning._evaluate(model, [{}, {}], ["a", "b"], torch)
        self.assertGreater(trained["samples_per_second"], 0)
        self.assertEqual(sum(map(sum, evaluated["confusion_matrix"])), 2)
        loaded = checkpoints._load_checkpoint(checkpoints._checkpoint_bytes(model, ["a", "b"], 0, torch), torch)
        self.assertTrue(all(t.device.type == "cpu" for t in loaded["state_dict"].values()))
        learning._fed_avg(model, [{"samples": 2, "checkpoint": loaded}], torch)
        self.assertEqual(next(model.parameters()).device.type, name)

    def test_fault_is_opt_in_job_scoped_one_shot_and_never_uses_cpu(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(devices, "GPU_FAULT_ROOT", Path(directory)),
            patch.dict(os.environ, {"ET_BENCHMARK_GPU_FAULTS": "1", "HARBOUR_FL_JOB_ID": "job"}),
        ):
            request = {"schema_version": 1, "job_id": "job", "mode": "runtime_error", "request_id": "request-1"}
            path = Path(directory, "request.json")
            path.write_text(json.dumps(request))
            with self.assertRaisesRegex(ValueError, "requires a GPU"):
                devices.check_gpu_fault(torch.device("cpu"), torch)
            with self.assertRaises(devices.GPUExecutionError):
                devices.check_gpu_fault(torch.device("cuda"), torch)
            devices.check_gpu_fault(torch.device("cuda"), torch)
            self.assertFalse(path.exists())
            self.assertEqual(json.loads(Path(directory, "receipt.json").read_text())["request_id"], "request-1")
            request.update(mode="out_of_memory", request_id="request-2")
            path.write_text(json.dumps(request))
            with self.assertRaises(torch.OutOfMemoryError):
                devices.check_gpu_fault(torch.device("mps"), torch)
            self.assertTrue(devices.gpu_error(torch.OutOfMemoryError("MPS out of memory"), torch.device("mps")))

    def test_disabled_and_foreign_requests_cannot_inject_a_failure(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(devices, "GPU_FAULT_ROOT", Path(directory)):
            request = {"schema_version": 1, "job_id": "foreign", "mode": "runtime_error", "request_id": "one"}
            path = Path(directory, "request.json")
            path.write_text(json.dumps(request))
            with patch.dict(os.environ, {"ET_BENCHMARK_GPU_FAULTS": "0"}):
                devices.check_gpu_fault(torch.device("cuda"), torch)
            self.assertTrue(path.exists())
            with patch.dict(os.environ, {"ET_BENCHMARK_GPU_FAULTS": "1", "HARBOUR_FL_JOB_ID": "job"}):
                devices.check_gpu_fault(torch.device("cuda"), torch)
            self.assertTrue(path.exists())

    def test_gpu_failure_stops_worker_and_requests_core_recovery_without_cpu_retry(self):
        from harbour_fl import client

        role = ClientRole()
        messages = queue.Queue()
        messages.put({"round": 0, "job_id": "job"})
        role.state = {"client_id": "vessel_1", "stop": threading.Event(), "update_queue": messages, "seen_rounds": {0}}
        role._event = Mock()
        role._mark_ready = Mock()
        role._send_client_status = Mock()
        role._send_json = Mock()
        role._run_client_round = Mock(side_effect=devices.GPUExecutionError("injected accelerator failure"))
        with (
            patch.object(client, "select_device", return_value=torch.device("cuda")),
            patch.object(client, "_data_root"),
            patch.object(client, "_labels", return_value=["a", "b"]),
            patch.object(client, "_load_rows", return_value=[{}]),
            self.assertLogs("harbour_fl", level="ERROR"),
        ):
            role._client_main()
        self.assertTrue(role.state["gpu_failed"])
        self.assertTrue(role._send_json.call_args.args[0]["recoverable"])
        role._run_client_round.assert_called_once()

    def test_coordinator_waits_for_replacement_without_discarding_the_round(self):
        from harbour_fl.coordinator import CoordinatorRole

        role = CoordinatorRole()
        messages = queue.Queue()
        messages.put(
            {
                "type": settings.UPDATE_TYPE,
                "job_id": "job",
                "round": 0,
                "client_id": "client",
                "status": "failed",
                "recoverable": True,
            }
        )
        messages.put(
            {
                "type": settings.UPDATE_TYPE,
                "job_id": "job",
                "round": 0,
                "client_id": "client",
                "status": "ready",
                "bucket": "client",
                "filename": "update.pt",
                "samples": 2,
            }
        )
        payload = checkpoints._checkpoint_bytes(torch.nn.Linear(2, 2), ["a", "b"], 0, torch)
        storage = Mock()
        storage.get.return_value = payload
        role.state = {"agent_id": "coordinator", "storage": storage, "update_queue": messages}
        role._event = Mock()
        role._send_json = Mock()
        role._send_coordinator_status = Mock()
        with patch.dict(os.environ, {"HARBOUR_FL_ROUND_TIMEOUT_SECONDS": "1"}):
            updates = role._collect_round("job", 0, "global.pt", ("client",))
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0]["status"], "ready")
        self.assertEqual(role._event.call_args_list[0].args[0], "client-recovery-pending")
