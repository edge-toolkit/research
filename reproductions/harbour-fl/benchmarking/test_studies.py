"""Recovery injections restore deployments and permissions, including failed and interrupted attempts."""

import copy
import json
import os
import runpy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from et_experiments.utils import BenchmarkError
from integration import HarbourFLAdapter
from recovery_faults import HarbourRecoveryFaults
from study_support import prepared_metadata


class RecoveryFaultTests(unittest.TestCase):
    def test_client_names_are_normalized_before_identity_validation(self):
        source = Path(__file__).resolve().parents[1] / "src/harbour_fl/settings.py"
        with patch.dict(os.environ, {"HARBOUR_FL_CLIENTS": "vessel_1, vessel_2,vessel_3 "}):
            self.assertEqual(runpy.run_path(str(source))["DEFAULT_CLIENTS"], ("vessel_1", "vessel_2", "vessel_3"))
        with patch.dict(os.environ, {"HARBOUR_FL_CLIENTS": "vessel_1, vessel_1"}), self.assertRaises(ValueError):
            runpy.run_path(str(source))

    def test_fault_and_cleanup_failures_are_both_preserved(self):
        self.before["restart_budget"]["restarts"]["vessel-3"] = 0
        self.faults.scheduling.restore.side_effect = BenchmarkError("cleanup denied")
        with self.assertRaisesRegex(BenchmarkError, "not observed.*cleanup denied"):
            self.faults.inject_fault({"type": "restart_command_rejected", "target": "vessel-3"}, "job")

    def test_cli_engine_uses_the_maintained_adapter_and_recovery_faults(self):
        from benchmark import engine

        instance = engine()
        self.assertIsInstance(instance.application, HarbourFLAdapter)
        self.assertIsInstance(instance.faults, HarbourRecoveryFaults)

    def setUp(self):
        self.adapter = HarbourFLAdapter()
        self.backend = Mock()
        self.before = {
            "context": {
                "schema_version": 1,
                "configuration": {"job_id": "job"},
                "jitter_seed": 7,
                "restored_sessions": 0,
                "elapsed_seconds": 10,
            },
            "restart_budget": {"restarts": {"vessel-3": 1}},
            "incidents": [
                {
                    "workload": "vessel-3",
                    "onset": 1,
                    "baseline": 0,
                    "detected": 2,
                    "recovered": None,
                    "actions": [{"kind": "restart-failed"}],
                }
            ],
        }
        self.after = copy.deepcopy(self.before)
        self.after["context"].update(restored_sessions=1, elapsed_seconds=11)
        self.after["clock_basis"] = "observer-elapsed-seconds-with-restart-gaps"
        self.after["observation_gaps"] = [{"reason": "supervision stopped", "started_at": 10, "ended_at": 11}]
        self.replaced = False

        def command(*args, **_kwargs):
            text = " ".join(args)
            value = {}
            if "get deployment/" in text:
                value = {
                    "spec": {
                        "replicas": 1,
                        "template": {
                            "spec": {
                                "containers": [{"volumeMounts": [{"name": "storage", "mountPath": "/app/storage"}]}],
                                "volumes": [{"name": "storage", "persistentVolumeClaim": {"claimName": "checkpoints"}}],
                            }
                        },
                    }
                }
            elif "get role/" in text:
                value = {
                    "rules": [{"resources": ["deployments"], "resourceNames": ["vessel-3"], "verbs": ["get", "patch"]}]
                }
            elif "rollout restart deployment/recovery-supervisor" in text:
                self.replaced = True
            elif "exec deployment/recovery-supervisor" in text:
                value = self.after if self.replaced else self.before
            return SimpleNamespace(stdout=json.dumps(value))

        self.backend.kubectl.side_effect = command
        self.faults = HarbourRecoveryFaults(self.backend, self.adapter)
        self.faults.scheduling = Mock()
        self.faults.scheduling.inject.return_value = {"status": "restored"}

    def test_hub_outage_restores_on_interrupt_and_scopes_namespace_before_exec(self):
        fault = {"type": "hub_outage", "target": "ws-server", "duration_seconds": 30}
        with patch("recovery_faults.time.sleep", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            self.faults.inject_fault(fault, "job")
        commands = [call.args for call in self.backend.kubectl.call_args_list]
        self.assertTrue(all(args[:2] == ("-n", self.adapter.config.namespace) for args in commands))
        self.assertTrue(any("--replicas=1" in args for args in commands))
        self.assertTrue(any("rollout" in args and "status" in args for args in commands))

    def test_rejected_restart_records_durable_budget_and_restores_permissions(self):
        result = self.faults.inject_fault(
            {"type": "restart_command_rejected", "target": "vessel-3", "duration_seconds": 70}, "job"
        )
        self.assertEqual(result["status"], "restored")
        self.assertTrue(result["durable_budget_verified"])
        self.assertEqual(result["budget_after"]["restart_budget"], self.before["restart_budget"])
        self.faults.scheduling.restore.assert_called_once()
        patches = [call.args for call in self.backend.kubectl.call_args_list if "patch" in call.args]
        self.assertEqual(json.loads(patches[-1][-1])[0]["path"], "/rules")
        executions = [call.args for call in self.backend.kubectl.call_args_list if "exec" in call.args]
        self.assertTrue(all(args.index("-n") < args.index("--") for args in executions))

    def test_missing_rejection_evidence_still_restores_permissions(self):
        self.before["restart_budget"]["restarts"]["vessel-3"] = 0
        with self.assertRaisesRegex(BenchmarkError, "not observed"):
            self.faults.inject_fault({"type": "restart_command_rejected", "target": "vessel-3"}, "job")
        self.faults.scheduling.restore.assert_called_once()

    def test_incomplete_fault_cleanup_is_retried_and_cannot_be_reported_as_restored(self):
        failed = Mock(side_effect=BenchmarkError("permission restoration failed"))
        remaining = Mock()
        self.faults._pending = [failed, remaining]
        with patch("integration.configure_run"):
            self.adapter.configure_run(self.backend, self.faults, "job", {}, None)
        with patch("integration.restore") as restore, self.assertRaisesRegex(BenchmarkError, "permission restoration"):
            self.adapter.restore(self.backend)
        restore.assert_called_once_with(self.backend)
        remaining.assert_called_once()
        failed.side_effect = None
        self.faults.restore_recovery()
        self.assertEqual(self.faults._pending, [])

    def test_invalid_faults_are_rejected_before_deployment_changes(self):
        for fault in (
            {"type": "hub_outage", "target": "vessel-3"},
            {"type": "restart_command_rejected", "target": "ws-server"},
            {"type": "hub_outage", "target": "ws-server", "duration_seconds": 0},
        ):
            with self.subTest(fault=fault), self.assertRaises(BenchmarkError):
                self.adapter.prepare_scenario({"rounds": 3, "fault": fault})
        self.backend.kubectl.assert_not_called()


class ScalingContractTests(unittest.TestCase):
    def test_current_adapter_and_runtime_settings_accept_twenty_four_clients(self):
        adapter = HarbourFLAdapter()
        workers = tuple(f"vessel-{i}" for i in range(1, 25))
        adapter.config = replace(adapter.config, workers=workers)
        scenario = adapter.prepare_scenario({"rounds": 1, "minimum_clients": 24, "fault": {"type": "none"}})
        self.assertEqual(scenario["model_profile"], "pretrained")
        logs = {
            worker: f"{worker.replace('-', '_')} completed round 0: {{'processed_samples': 2}}" for worker in workers
        }
        self.assertEqual(len(adapter.parse_logs(logs, 1)["client_rounds"]), 24)
        path = Path(__file__).parents[1] / "src/harbour_fl/settings.py"
        names = ",".join(worker.replace("-", "_") for worker in workers)
        with patch.dict(os.environ, {"HARBOUR_FL_CLIENTS": names}):
            self.assertEqual(len(runpy.run_path(str(path))["DEFAULT_CLIENTS"]), 24)
        with patch.dict(os.environ, {"HARBOUR_FL_CLIENTS": "vessel_1,vessel_1"}), self.assertRaises(ValueError):
            runpy.run_path(str(path))

    def test_generated_partition_metadata_passes_the_current_loader_and_rejects_tampering(self):
        from harbour_fl.pretrained_assets import fingerprint
        from harbour_fl.pretrained_data import load_rows
        from harbour_fl.provenance import file_hash

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature = root / "feature.npz"
            feature.write_bytes(b"unchanged-feature-archive")
            info = {
                "input": {"pretrained_assets": fingerprint()},
                "feature_dimension": 960,
                "files": {"feature.npz": file_hash(feature)},
                "class_weights": [1, 2],
                "labels": ["a", "b"],
            }
            manifest = "clients/vessel_24/train_manifest.csv"
            data = {manifest: "segment_id,standard_label,feature_path\ns1,a,/scaling-source/feature.npz\n"}
            for name, text in data.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            (root / "preparation.json").write_text(prepared_metadata(info, data))

            def mounted_hash(path):
                return file_hash(feature if str(path) == "/scaling-source/feature.npz" else path)

            with patch("harbour_fl.pretrained_data.file_hash", side_effect=mounted_hash):
                loaded = load_rows(root, root / manifest)
                self.assertEqual(loaded[0]["class_weights"], info["class_weights"])
                self.assertEqual(loaded[0]["segment_id"], "s1")
                (root / manifest).write_text(data[manifest] + "tampered")
                with self.assertRaisesRegex(ValueError, "manifest changed"):
                    load_rows(root, root / manifest)
