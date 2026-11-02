"""Pinned assets, head-only federated training, portable features and explicit missing-modality masks."""

import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from harbour_fl import checkpoints, data, learning, pretrained_assets, pretrained_data
from harbour_fl.pretrained_model import Fusion
from harbour_fl.provenance import digest, file_hash, save
from harbour_fl.workload import describe, signature
from test_fixtures import prepared


class PretrainedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.environment = patch.dict(
            os.environ,
            {
                "HARBOUR_FL_MODEL_PROFILE": "pretrained",
                "HARBOUR_FL_ADAPTATION": "false",
                "HARBOUR_FL_DETECTOR_ENABLED": "false",
                "HARBOUR_FL_MISSING_MODALITY": "none",
                "HARBOUR_FL_AUDIO_NOISE_STD": "0",
                "HARBOUR_FL_IMAGE_NOISE_STD": "0",
                "HARBOUR_FL_LOCAL_EPOCHS": "1",
                "HARBOUR_FL_BATCH_SIZE": "2",
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_assets_fail_closed_and_import_requires_recorded_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            source.mkdir()
            (source / "model.pt").write_bytes(b"trusted checkpoint")
            record = {"files": {"model.pt": {"sha256": hashlib.sha256(b"trusted checkpoint").hexdigest()}}}
            with patch.object(pretrained_assets, "identity", return_value=record):
                with self.assertRaisesRegex(ValueError, "missing or changed"):
                    pretrained_assets.verify(output)
                pretrained_assets.prepare_assets(output, source)
                self.assertEqual(pretrained_assets.verify(output), record)
                (output / "model.pt").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "missing or changed"):
                    pretrained_assets.verify(output)
                (source / "model.pt").write_bytes(b"wrong")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    pretrained_assets.prepare_assets(output, source)

    def test_pinned_upstream_download_verifies_source_before_namespacing(self):
        original = b"from models.mn.utils import value\n"
        converted = b"from .utils import value\n"
        document = {
            "efficientat_repository": "https://example.invalid/source/",
            "efficientat_revision": "pinned",
            "files": {
                "efficientat/model.py": {
                    "upstream_path": "models/mn/model.py",
                    "upstream_sha256": hashlib.sha256(original).hexdigest(),
                    "sha256": hashlib.sha256(converted).hexdigest(),
                }
            },
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(pretrained_assets, "identity", return_value=document),
        ):
            root = Path(directory)
            with patch("urllib.request.urlopen", return_value=io.BytesIO(original)) as download:
                pretrained_assets.prepare_assets(root)
            download.assert_called_once_with("https://example.invalid/source/pinned/models/mn/model.py", timeout=120)
            self.assertEqual((root / "efficientat/model.py").read_bytes(), converted)
            (root / "efficientat/model.py").unlink()
            with (
                patch("urllib.request.urlopen", return_value=io.BytesIO(b"changed upstream source")),
                self.assertRaisesRegex(ValueError, "upstream source hash mismatch"),
            ):
                pretrained_assets.prepare_assets(root)

    def test_mask_after_layernorm_and_presence_flags_match_unavailable_inputs(self):
        model = Fusion(2).eval()
        model.audio_norm.bias.data.fill_(4)
        captured = []
        handle = model.classifier.register_forward_pre_hook(lambda _module, values: captured.append(values[0]))
        try:
            output = model(torch.zeros(2, 960), torch.rand(2, 960))
            self.assertEqual(tuple(output.shape), (2, 2))
            self.assertEqual(int(torch.count_nonzero(captured[0][:, :960])), 0)
            torch.testing.assert_close(captured[0][:, -2:], torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
        finally:
            handle.remove()

    def test_real_head_training_aggregation_and_checkpoint_resume(self):
        """Verify real head training aggregation and checkpoint resume."""
        with tempfile.TemporaryDirectory() as directory:
            root = prepared(Path(directory))
            rows = data._load_rows(root, root / "test_manifest.csv")
            model = learning._new_model(torch, 2)
            before = {key: value.clone() for key, value in model.state_dict().items()}
            metrics = learning._train_local(model, rows, ["a", "b"], 0, torch)
            self.assertEqual(metrics["optimizer_steps"], 1)
            self.assertTrue(any(not torch.equal(before[key], value) for key, value in model.state_dict().items()))
            self.assertFalse(any("features" in key for key in model.state_dict()))
            payload = checkpoints._checkpoint_bytes(model, ["a", "b"], 1, torch)
            saved = checkpoints._load_checkpoint(payload, torch)
            replacement = learning._new_model(torch, 2)
            replacement.load_state_dict(saved["state_dict"], strict=True)
            learning._fed_avg(replacement, [{"samples": 2, "checkpoint": saved}], torch)
            for key, value in replacement.state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[key])
            self.assertEqual(describe(model, torch, torch.device("cpu"))["optimizer"], "adamw")
            self.assertEqual(saved["workload"], signature())
            with (
                patch.dict(os.environ, {"HARBOUR_FL_MODEL_PROFILE": "unsupported"}),
                self.assertRaisesRegex(ValueError, "MODEL_PROFILE"),
            ):
                checkpoints._load_checkpoint(payload, torch)

    def test_cached_conditions_missing_modalities_and_legacy_audio_rejection(self):
        """Verify cached conditions missing modalities and legacy audio rejection."""
        with tempfile.TemporaryDirectory() as directory:
            root = prepared(Path(directory))
            rows = data._load_rows(root, root / "test_manifest.csv")
            clean = data._materialize(rows, ["a", "b"], torch)
            self.assertEqual(tuple(clean[0].shape), (2, 960))
            for name, index in (("audio", 0), ("image", 1)):
                values = data._materialize(rows, ["a", "b"], torch, {"missing_modality": name})
                self.assertEqual(int(torch.count_nonzero(values[index])), 0)
                torch.testing.assert_close(values[1 - index], clean[1 - index])
            for condition, audio_offset, image_offset in (
                ({"audio_snr_db": 20}, 1, 0),
                ({"audio_snr_db": 10}, 2, 0),
                ({"audio_snr_db": 0}, 3, 0),
                ({"image_noise_std": 0.05}, 0, 4),
                ({"image_noise_std": 0.1}, 0, 5),
                ({"image_noise_std": 0.2}, 0, 6),
            ):
                with self.subTest(condition=condition):
                    audio, images, targets = data._materialize(rows, ["a", "b"], torch, condition)
                    torch.testing.assert_close(audio, clean[0] + audio_offset)
                    torch.testing.assert_close(images, clean[1] + image_offset)
                    torch.testing.assert_close(targets, clean[2])
            with self.assertRaisesRegex(ValueError, "waveform SNR"):
                data._materialize(rows, ["a", "b"], torch, {"audio_noise_std": 0.5})
            with self.assertRaisesRegex(ValueError, "simultaneous"):
                data._materialize(rows, ["a", "b"], torch, {"audio_snr_db": 10, "image_noise_std": 0.1})
            (root / "sample.npz").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "archive changed"):
                data._load_rows(root, root / "test_manifest.csv")

    def test_augmentation_preserves_configured_faults_and_randomizes_only_compatible_inputs(self):
        """Keep fixed noise and missing-input faults visible in the actual tensors used for local fitting."""
        cases = (
            ({}, 0, 5, True),
            ({"HARBOUR_FL_MISSING_MODALITY": "audio"}, None, 5, True),
            ({"HARBOUR_FL_IMAGE_NOISE_STD": "0.2"}, 0, 6, False),
            ({"HARBOUR_FL_AUDIO_SNR_DB": "0"}, 3, 0, False),
            ({"HARBOUR_FL_AUDIO_SNR_DB": "10", "HARBOUR_FL_MISSING_MODALITY": "image"}, 2, None, False),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = prepared(Path(directory))
            rows = data._load_rows(root, root / "test_manifest.csv")
            clean = data._materialize(rows, ["a", "b"], torch, {})
            for changes, audio_offset, image_offset, randomized in cases:
                with (
                    self.subTest(changes=changes),
                    patch.dict(os.environ, {"HARBOUR_FL_ADAPTATION": "true", **changes}),
                ):
                    if "HARBOUR_FL_AUDIO_SNR_DB" not in changes:
                        os.environ.pop("HARBOUR_FL_AUDIO_SNR_DB", None)
                    model = Fusion(2)
                    captured = []
                    hook = model.register_forward_pre_hook(
                        lambda _model, values, captured=captured: captured.append(
                            tuple(value.detach().clone() for value in values)
                        )
                    )
                    try:
                        with patch.object(torch, "randint", return_value=torch.tensor(2)) as choose:
                            learning._train_local(model, rows, ["a", "b"], 0, torch)
                        self.assertEqual(choose.call_count, int(randomized))
                        self.assertEqual(len(captured), 1)
                        for actual, reference, offset in zip(
                            captured[0], clean[:2], (audio_offset, image_offset), strict=True
                        ):
                            expected = torch.zeros_like(reference) if offset is None else reference + offset
                            torch.testing.assert_close(actual, expected)
                    finally:
                        hook.remove()

    def test_augmentation_does_not_hide_invalid_or_unprepared_noise_faults(self):
        """Reject invalid configured corruption rather than replacing it with a supported random condition."""
        cases = (
            ({"HARBOUR_FL_AUDIO_NOISE_STD": "0.1"}, "waveform SNR"),
            ({"HARBOUR_FL_AUDIO_SNR_DB": "5"}, "prepared waveform SNR"),
            ({"HARBOUR_FL_AUDIO_SNR_DB": "10", "HARBOUR_FL_IMAGE_NOISE_STD": "0.2"}, "simultaneous"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = prepared(Path(directory))
            rows = data._load_rows(root, root / "test_manifest.csv")
            for changes, reason in cases:
                with (
                    self.subTest(changes=changes),
                    patch.dict(os.environ, {"HARBOUR_FL_ADAPTATION": "true", **changes}),
                ):
                    if "HARBOUR_FL_AUDIO_SNR_DB" not in changes:
                        os.environ.pop("HARBOUR_FL_AUDIO_SNR_DB", None)
                    with patch.object(torch, "randint") as choose, self.assertRaisesRegex(ValueError, reason):
                        learning._train_local(Fusion(2), rows, ["a", "b"], 0, torch)
                    choose.assert_not_called()

    def test_pretrained_profile_rejects_old_features_and_modified_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "prepared frozen features"):
                pretrained_data.metadata(root)
            prepared(root)
            doc = json.loads((root / "preparation.json").read_text())
            doc["feature_dimension"] = 64
            save(root / "preparation.json", doc)
            with self.assertRaisesRegex(ValueError, "metadata changed"):
                pretrained_data.metadata(root)

    def test_class_summary_hash_and_label_order_are_verified_before_use(self):
        """Verify class summary hash and label order are verified before use."""
        with tempfile.TemporaryDirectory() as directory:
            root = prepared(Path(directory))
            self.assertEqual(data._labels(root), ["a", "b"])
            summary = root / "class_summary.csv"
            summary.write_text("standard_label,segments\nb,1\na,1\n")
            with self.assertRaisesRegex(ValueError, "class summary changed"):
                data._labels(root)
            marker = root / "preparation.json"
            record = json.loads(marker.read_text())
            record["files"][summary.name] = file_hash(summary)
            record["fingerprint"] = digest({key: value for key, value in record.items() if key != "fingerprint"})
            save(marker, record)
            with self.assertRaisesRegex(ValueError, "class labels do not match"):
                data._labels(root)
