"""Raw-input compatibility, source filtering and recording/content-disjoint preparation."""

import csv
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from harbour_fl.prepare_dataset import load_dataset, prepare, resolve_audio
from harbour_fl.pretrained_inputs import AudioFrontend, decode_audio, sample_seed


class PreparationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_missing_client_training_coverage_fails_before_saving_or_encoding(self):
        """Verify missing client training coverage fails before saving or encoding."""
        from harbour_fl.prepare import encode

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for label in ("passenger", "sailing"):
                for number in range(3):
                    event = f"{label}-{number}"
                    audio, image = root / f"{event}.wav", root / f"{event}.jpg"
                    audio.write_bytes(event.encode())
                    image.write_bytes(("image-" + event).encode())
                    rows.append(
                        {
                            "label": label,
                            "event_id": event,
                            "segment_id": event,
                            "path": str(audio),
                            "camera_image_output_path": image.name,
                            "segment_start_sec": 0,
                            "segment_end_sec": 1,
                        }
                    )
            with (root / "segment_manifest.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "no training samples"):
                prepare(root, root / "mapped")
            self.assertFalse((root / "mapped/dataset.json").exists())
            with (
                patch("harbour_fl.prepare.Encoders") as encoder,
                self.assertRaisesRegex(ValueError, "no training samples"),
            ):
                encode({"rows": [{"split": "train", "client": "client-1"}]}, root / "encoded", root, "cpu")
            encoder.assert_not_called()
            self.assertFalse((root / "encoded").exists())

    def test_mps_encoder_placement_keeps_frontend_on_cpu(self):
        from unittest.mock import Mock

        from harbour_fl.pretrained_model import Encoders

        encoder = Encoders.__new__(Encoders)
        torch.nn.Module.__init__(encoder)
        encoder.audio, encoder.image, encoder.frontend = Mock(), Mock(), Mock()
        self.assertIs(encoder.place("mps"), encoder)
        encoder.audio.to.assert_called_once_with(torch.device("mps"))
        encoder.image.to.assert_called_once_with(torch.device("mps"))
        encoder.frontend.to.assert_called_once_with("cpu")

    def test_recording_split_and_rare_class_exclusion(self):
        """Verify recording split and rare class exclusion."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for label, count in (("passenger", 6), ("sailing", 6), ("fishing", 2)):
                for number in range(count):
                    event = f"{label}-{number}"
                    audio, image = root / (event + ".wav"), root / (event + ".jpg")
                    audio.write_bytes(event.encode())
                    image.write_bytes(("image" + event).encode())
                    for segment in range(2):
                        rows.append(
                            {
                                "label": label,
                                "standard_label": label,
                                "event_id": event,
                                "segment_id": f"{event}-{segment}",
                                "path": str(audio),
                                "camera_image_output_path": image.name,
                                "segment_start_sec": segment,
                                "segment_end_sec": segment + 1,
                            }
                        )
            with (root / "segment_manifest.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = prepare(root, root / "mapped")
            self.assertEqual(result["labels"], ["passenger", "sailing"])
            self.assertEqual(len(result["excluded"]), 4)
            for key in ("event_id", "audio_sha256", "image_sha256"):
                split_sets = [
                    {r[key] for r in result["rows"] if r["split"] == split} for split in ("train", "val", "test")
                ]
                self.assertFalse(
                    split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2]
                )
            self.assertEqual(load_dataset(root / "mapped", verify_files=True)["fingerprint"], result["fingerprint"])
            Path(result["rows"][0]["audio"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "source asset changed"):
                load_dataset(root / "mapped", verify_files=True)

    def test_pcm24_decoding_and_pretrained_frontend_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audio.wav"
            rate = 48000
            samples = np.rint(np.sin(np.arange(rate * 2) / rate * 2 * np.pi * 440) * 0.1 * 8388607).astype(np.int32)
            packed = np.stack([samples & 255, (samples >> 8) & 255, (samples >> 16) & 255], axis=1).astype(np.uint8)
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(3)
                stream.setframerate(rate)
                stream.writeframes(packed.tobytes())
            audio = decode_audio({"audio": str(path), "start": 0.5, "end": 1.5, "segment_id": "x"})
            self.assertEqual(tuple(audio.shape), (32000,))
            self.assertAlmostEqual(audio.square().mean().sqrt().item(), 0.1 / np.sqrt(2), places=3)
            features = AudioFrontend()(audio[None])
            self.assertEqual(tuple(features.shape), (1, 1, 128, 100))
            self.assertTrue(torch.isfinite(features).all())
            self.assertEqual(sample_seed("x", "audio-snr10", 17), sample_seed("x", "audio-snr10", 17))
            self.assertNotEqual(sample_seed("x", "audio-snr10", 17), sample_seed("x", "audio-snr0", 17))

    def test_mixed_manifest_filters_before_labels_and_resolves_raw_camera_paths(self):
        """Verify mixed manifest filters before labels and resolves raw camera paths."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, manifests = root / "raw", root / "mixed"
            manifests.mkdir()
            rows = []
            for label in ("passenger", "sailing"):
                for i in range(4):
                    event = f"{label}-{i}"
                    folder = raw / label / event
                    folder.mkdir(parents=True)
                    (folder / "record.wav").write_bytes(event.encode())
                    (folder / "Camera.JPG").write_bytes(("image-" + event).encode())
                    previous = Path("/another/computer") / label / event
                    rows.append(
                        {
                            "source_dataset": "Croatia",
                            "label": label,
                            "event_id": event,
                            "segment_id": event,
                            "resolved_audio_path": str(previous / "record.wav"),
                            "camera_image_path": str(previous / "Camera.JPG"),
                            "segment_start_sec": 0,
                            "segment_end_sec": 5,
                        }
                    )
            onc = {**rows[0], "source_dataset": "ONC_VTUAD", "label": "background"}
            unknown = {**rows[0], "source_dataset": "", "label": "unmapped"}
            rows = [onc, unknown, *rows]
            source = manifests / "segment_manifest.csv"
            with source.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            before = source.read_bytes()
            value = prepare(manifests, root / "prepared", raw_root=raw)
            self.assertEqual(len(value["rows"]), 8)
            self.assertEqual(value["import_summary"]["manifest_rows"], 10)
            self.assertEqual(value["import_summary"]["skipped_other_source_rows"], 2)
            self.assertTrue(all(Path(row["image"]).is_relative_to(raw.resolve()) for row in value["rows"]))
            self.assertTrue(all(row["source_dataset"] == "Croatia" for row in value["rows"]))
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(prepare(manifests, root / "prepared", raw_root=raw)["fingerprint"], value["fingerprint"])

    def test_explicit_raw_root_does_not_fall_back_to_another_dataset_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old/class/event/audio.wav"
            old.parent.mkdir(parents=True)
            old.write_bytes(b"original")
            with self.assertRaisesRegex(ValueError, "raw WAV unavailable"):
                resolve_audio({"path": str(old), "segment_id": "x"}, root / "missing")

    def test_nested_recording_paths_preserve_vessel_subfolders(self):
        from harbour_fl.prepare_dataset import resolve_image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "Ferries/BlueWhite_Ferry_Postira/recording-event"
            folder.mkdir(parents=True)
            (folder / "audio.wav").write_bytes(b"audio")
            (folder / "camera.JPG").write_bytes(b"image")
            previous = Path("/old/machine/downloads/V1/Ferries/BlueWhite_Ferry_Postira/recording-event")
            row = {
                "path": str(previous / "audio.wav"),
                "camera_image_path": str(previous / "camera.JPG"),
                "segment_id": "example",
            }
            self.assertEqual(resolve_audio(row, root), (folder / "audio.wav").resolve())
            self.assertEqual(resolve_image(row, root / "manifests", root), (folder / "camera.JPG").resolve())

    def test_manifest_without_croatia_records_stops_clearly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "segment_manifest.csv").write_text("source_dataset,segment_id,label\nONC_VTUAD,x,background\n")
            with self.assertRaisesRegex(ValueError, "no Croatia records"):
                prepare(root, root / "prepared")

    def test_reused_ids_are_qualified_and_identical_pairs_are_not_double_counted(self):
        """Verify reused ids are qualified and identical pairs are not double counted."""
        for mode in ("different-recordings", "same-content", "conflicting-labels"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                rows = []
                for label in ("passenger", "sailing"):
                    for i in range(4):
                        event = f"{label}-{i}"
                        audio, image = root / f"{event}.wav", root / f"{event}.jpg"
                        duplicate = (mode == "same-content" and label == "passenger" and i == 1) or (
                            mode == "conflicting-labels" and label == "sailing" and i == 0
                        )
                        content = "passenger-0" if duplicate else event
                        audio.write_bytes(content.encode())
                        image.write_bytes(("image-" + content).encode())
                        rows.append(
                            {
                                "label": label,
                                "event_id": event,
                                "segment_id": "reused" if label == "passenger" and i < 2 else event,
                                "path": str(audio),
                                "camera_image_output_path": image.name,
                                "segment_start_sec": 0,
                                "segment_end_sec": 5,
                            }
                        )
                with (root / "segment_manifest.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                if mode == "conflicting-labels":
                    with self.assertRaisesRegex(ValueError, "identical paired segment has conflicting labels"):
                        prepare(root, root / "prepared")
                    continue
                result = prepare(root, root / "prepared")
                self.assertEqual(len(result["rows"]), 7 if mode == "same-content" else 8)
                self.assertEqual(len({r["segment_id"] for r in result["rows"]}), len(result["rows"]))
                self.assertEqual(result["import_summary"]["duplicate_paired_segments"], int(mode == "same-content"))

    def test_same_image_connects_recordings_before_splitting(self):
        # Reuse the production preparation path with two source events sharing image bytes.
        """Verify same image connects recordings before splitting."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for label in ("passenger", "sailing"):
                for i in range(6):
                    audio, image = root / f"{label}-{i}.wav", root / f"{label}-{i}.jpg"
                    audio.write_bytes(f"audio-{label}-{i}".encode())
                    image.write_bytes(f"image-{label}-{0 if i == 5 else i}".encode())
                    rows.append(
                        {
                            "label": label,
                            "event_id": f"{label}-{i}",
                            "segment_id": f"{label}-{i}",
                            "path": str(audio),
                            "camera_image_output_path": image.name,
                            "segment_start_sec": 0,
                            "segment_end_sec": 1,
                        }
                    )
            with (root / "segment_manifest.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            value = prepare(root, root / "prepared")
            self.assertEqual(value["independent_groups_per_class"], {"passenger": 5, "sailing": 5})
            for label in ("passenger", "sailing"):
                merged = [r for r in value["rows"] if r["event_id"] in {f"{label}-0", f"{label}-5"}]
                self.assertEqual(len({r["split"] for r in merged}), 1)
