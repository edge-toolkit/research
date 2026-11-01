"""One pretrained configuration, strict checkpoint identity and real native federated head training."""

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from harbour_fl import calibrate, checkpoints, data, learning
from harbour_fl.calibrate import inventory, measure
from harbour_fl.workload import describe, signature, workload
from test_fixtures import prepared


class WorkloadTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.environment = patch.dict(
            os.environ,
            {
                "HARBOUR_FL_MODEL_PROFILE": "pretrained",
                "HARBOUR_FL_DETECTOR_ENABLED": "false",
                "HARBOUR_FL_LOCAL_EPOCHS": "1",
                "HARBOUR_FL_BATCH_SIZE": "2",
                "HARBOUR_FL_ADAPTATION": "false",
                "HARBOUR_FL_MISSING_MODALITY": "none",
                "HARBOUR_FL_AUDIO_NOISE_STD": "0",
                "HARBOUR_FL_IMAGE_NOISE_STD": "0",
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_default_model_contains_only_the_trainable_fusion_head(self):
        """Verify default model contains only the trainable fusion head."""
        model = learning._new_model(torch, 2)
        self.assertEqual(workload().model_profile, "pretrained")
        self.assertEqual(model.classifier[0].in_features, 1922)
        self.assertEqual(model.audio_norm.normalized_shape, (960,))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertFalse(any("features" in key for key in model.state_dict()))
        info = describe(model, torch, torch.device("cpu"))
        self.assertEqual(info["trainable_scope"], "fusion-head-only-frozen-cached-encoders")
        self.assertEqual(info["parameters"], info["trainable_parameters"])
        self.assertEqual(info["optimizer"], "adamw")

    def test_retired_or_unknown_profiles_are_rejected(self):
        """Verify retired or unknown profiles are rejected."""
        for name in ("demo", "reference", "unknown"):
            with self.subTest(profile=name), self.assertRaisesRegex(ValueError, "MODEL_PROFILE"):
                workload(name)

    def test_checkpoint_requires_current_workload_identity(self):
        """Verify checkpoint requires current workload identity."""
        model = learning._new_model(torch, 2)
        payload = checkpoints._checkpoint_bytes(model, ["a", "b"], 0, torch)
        current = checkpoints._load_checkpoint(payload, torch)
        self.assertEqual(current["workload"], signature())
        for change in (None, {"model_profile": "retired"}, {**signature(), "pretrained_assets": "different"}):
            stale = {**current, "workload": change}
            output = io.BytesIO()
            torch.save(stale, output)
            with self.subTest(workload=change), self.assertRaisesRegex(ValueError, "workload"):
                checkpoints._load_checkpoint(output.getvalue(), torch)

    def test_native_calibration_trains_three_clients_and_evaluates_disjoint_partitions(self):
        """Verify native calibration trains three clients and evaluates disjoint partitions."""
        with tempfile.TemporaryDirectory() as directory:
            root = prepared(Path(directory))
            clients, test, metadata = inventory(root, ["a", "b"])
            model, records = measure(torch, torch.device("cpu"), clients, test, ["a", "b"], 1)
            self.assertEqual(len(records[0]["clients"]), 3)
            self.assertEqual(metadata["manifests"]["clients/vessel_1/train_manifest.csv"]["usable_samples"], 2)
            self.assertTrue(all(client["optimizer_steps"] == 1 for client in records[0]["clients"]))
            self.assertEqual(
                records[0]["checkpoint_bytes"], len(checkpoints._checkpoint_bytes(model, ["a", "b"], 1, torch))
            )
            audio, image, _targets = data._materialize(test, ["a", "b"], torch)
            self.assertEqual((tuple(audio.shape), tuple(image.shape)), ((2, 960), (2, 960)))

    def test_calibration_report_identifies_training_augmentation_and_waveform_noise(self):
        with tempfile.TemporaryDirectory() as directory:
            root = prepared(Path(directory))
            output = root / "calibration.json"
            with (
                patch.dict(
                    os.environ,
                    {
                        "HARBOUR_FL_DATA_ROOT": str(root),
                        "HARBOUR_FL_ADAPTATION": "true",
                        "HARBOUR_FL_AUDIO_SNR_DB": "10",
                    },
                ),
                patch("sys.argv", ["calibrate", "--output", str(output)]),
            ):
                self.assertEqual(calibrate.main(), 0)
            report = json.loads(output.read_text())
            self.assertTrue(report["adaptation"])
            self.assertEqual(report["perturbations"]["HARBOUR_FL_AUDIO_SNR_DB"], "10")
            self.assertEqual(len(report["rounds"][0]["clients"]), 3)
