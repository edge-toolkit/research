"""Frozen camera detection, safe crops, cache identity and the classifier input boundary."""

import dataclasses
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from harbour_fl.vision.config import detector_config, load_config, preprocessing_signature
from harbour_fl.vision.preprocessing import CameraPreprocessor
from harbour_fl.vision.yolo import YoloDetector, select_crop
from PIL import Image, ImageDraw


def configuration(root, **overrides):
    """Write a detector configuration and placeholder model with selected test overrides."""
    model = root / "model.onnx"
    if not model.exists():
        model.write_bytes(b"model-fixture")
    path = root / ("config-" + str(len(list(root.glob("config-*.json")))) + ".json")
    path.write_text(
        json.dumps(
            {"schema_version": 1, "model": "model.onnx", "num_classes": 1, "input_size": 32, "margin": 0, **overrides}
        )
    )
    return path


class ConfigTests(unittest.TestCase):
    def test_default_pipeline_is_disabled_without_loading_a_model(self):
        with patch.dict(
            os.environ, {"HARBOUR_FL_DETECTOR_ENABLED": "false", "HARBOUR_FL_DETECTOR_CONFIG": "missing.json"}
        ):
            self.assertIsNone(detector_config())
            self.assertEqual(preprocessing_signature(), {"enabled": False})

    def test_config_identifies_weights_and_validates_class_mapping(self):
        """Verify config identifies weights and validates class mapping."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_config(configuration(root, num_classes=3, class_ids=[2, 0, 2]))
            self.assertEqual(config.class_ids, (0, 2))
            self.assertEqual(config.model_sha256, hashlib.sha256(b"model-fixture").hexdigest())
            for invalid in (
                {"class_ids": [3]},
                {"confidence": float("nan")},
                {"input_size": 33},
                {"no_detection": "guess"},
                {"cache": 1},
                {"sha256": "wrong"},
                {"model": "missing.onnx"},
            ):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    load_config(configuration(root, **invalid))

    def test_cache_choice_does_not_change_semantic_checkpoint_signature(self):
        """Verify cache choice does not change semantic checkpoint signature."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signatures = []
            for cache in (True, False):
                path = configuration(root, cache=cache)
                with patch.dict(
                    os.environ, {"HARBOUR_FL_DETECTOR_ENABLED": "true", "HARBOUR_FL_DETECTOR_CONFIG": str(path)}
                ):
                    signatures.append(preprocessing_signature())
            self.assertEqual(*signatures)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is supplied by the optional detection runtime")
class DecodeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = load_config(configuration(Path(self.directory.name)))

    def output(self, rows):
        import numpy as np

        return np.array(rows, dtype=np.float32).T[None]

    def test_letterbox_coordinates_are_mapped_to_original_image(self):
        output = self.output([[16, 16, 16, 8, 0.9]])
        result = select_crop(output, self.config, (64, 32), (0.5, 0.5), (0, 8))
        self.assertEqual(result["box"], [16, 8, 48, 24])

    def test_non_vessel_classes_and_low_scores_are_filtered(self):
        config = dataclasses.replace(self.config, num_classes=2)
        output = self.output([[16, 16, 10, 10, 0.7, 0.9], [16, 16, 8, 8, 0.1, 0.05]])
        self.assertIsNone(select_crop(output, config, (32, 32), (1, 1), (0, 0))["box"])

    def test_largest_vessel_and_highest_confidence_are_explicit_choices(self):
        output = self.output([[16, 16, 20, 20, 0.6], [5, 5, 4, 4, 0.95]])
        largest = select_crop(output, self.config, (32, 32), (1, 1), (0, 0))
        confident = select_crop(
            output, dataclasses.replace(self.config, selection="confidence"), (32, 32), (1, 1), (0, 0)
        )
        self.assertEqual(largest["box"], [6, 6, 26, 26])
        self.assertEqual(confident["box"], [3, 3, 7, 7])

    def test_margin_is_clipped_and_duplicate_boxes_are_suppressed(self):
        output = self.output([[10, 10, 20, 20, 0.9], [10, 10, 19, 19, 0.8]])
        result = select_crop(output, dataclasses.replace(self.config, margin=0.5), (32, 32), (1, 1), (0, 0))
        self.assertEqual(result["box"], [0, 0, 30, 30])
        self.assertAlmostEqual(result["confidence"], 0.9, places=5)

    def test_invalid_outputs_fail_instead_of_becoming_missing_detections(self):
        """Verify invalid outputs fail instead of becoming missing detections."""
        for output in (
            self.output([[16, 16, 10, 10, float("nan")]]),
            self.output([[16, 16, 10, 10, 2]]),
            self.output([[0, 0, 1, 1, 0.9, 0]]),
        ):
            with self.assertRaises(ValueError):
                select_crop(output, self.config, (32, 32), (1, 1), (0, 0))


class CropTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = load_config(configuration(self.root))
        self.environment = patch.dict(os.environ, {"HARBOUR_FL_DETECTOR_CACHE": str(self.root / "cache")})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_cached_box_is_reused_and_source_pixels_remain_unchanged(self):
        fake = Mock()
        fake.predict.return_value = {"box": [2, 2, 8, 8], "confidence": 0.9, "class_id": 0}
        processor = CameraPreprocessor(self.config, fake)
        image = Image.new("RGB", (10, 10), "blue")
        before = image.tobytes()
        crop, first = processor.process(image)
        _, second = processor.process(image)
        self.assertEqual(crop.size, (6, 6))
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        fake.predict.assert_called_once()
        self.assertEqual(image.tobytes(), before)
        processor.process(Image.new("RGB", (10, 10), "red"))
        self.assertEqual(fake.predict.call_count, 2)

    def test_malformed_cache_is_recomputed(self):
        fake = Mock()
        fake.predict.return_value = {"box": [1, 1, 9, 9], "confidence": 0.9, "class_id": 0}
        processor = CameraPreprocessor(self.config, fake)
        image = Image.new("RGB", (10, 10))
        processor.process(image)
        next((self.root / "cache").rglob("*.json")).write_text("{}")
        _, result = processor.process(image)
        self.assertFalse(result["cache_hit"])
        self.assertEqual(fake.predict.call_count, 2)

    def test_missing_detection_policies_preserve_sample_count(self):
        """Verify missing detection policies preserve sample count."""
        fake = Mock()
        fake.predict.return_value = {"box": None, "confidence": None, "class_id": None}
        image = Image.new("RGB", (10, 10), "red")
        for policy in ("full-image", "zero"):
            crop, result = CameraPreprocessor(dataclasses.replace(self.config, no_detection=policy), fake).process(
                image
            )
            self.assertEqual(crop.size, image.size)
            self.assertEqual(result["outcome"], policy)
            self.assertEqual(crop.getpixel((0, 0)), (0, 0, 0) if policy == "zero" else (255, 0, 0))
        with self.assertRaisesRegex(ValueError, "no vessel detected"):
            CameraPreprocessor(dataclasses.replace(self.config, no_detection="error"), fake).process(image)

    def test_classifier_receives_the_crop_as_a_64_pixel_tensor(self):
        from harbour_fl.vision import crop_camera

        image = Image.new("RGB", (10, 10), "blue")
        ImageDraw.Draw(image).rectangle((2, 2, 7, 7), fill="red")
        path = self.root / "input.png"
        image.save(path)
        fake = Mock()
        fake.predict.return_value = {"box": [2, 2, 8, 8], "confidence": 0.9, "class_id": 0}
        with (
            patch.dict(
                os.environ,
                {"HARBOUR_FL_DETECTOR_ENABLED": "1", "HARBOUR_FL_DETECTOR_CONFIG": str(self.root / "config-0.json")},
            ),
            patch("harbour_fl.vision.preprocessing.processor", return_value=CameraPreprocessor(self.config, fake)),
        ):
            crop, _metadata = crop_camera(image)
        self.assertEqual(crop.size, (6, 6))
        self.assertEqual(set(crop.get_flattened_data()), {(255, 0, 0)})

    def test_client_rejects_coordinator_preprocessing_mismatch_before_training(self):
        from harbour_fl.runtime import VesselRuntime

        runtime = VesselRuntime()
        runtime.state["storage"] = Mock()
        runtime.state["storage"].get.return_value = b"checkpoint"
        with (
            patch("harbour_fl.client._load_checkpoint", return_value={"image_preprocessing": {"enabled": False}}),
            patch("harbour_fl.client.preprocessing_signature", return_value={"enabled": True}),
            self.assertRaisesRegex(ValueError, "preprocessing do not match"),
        ):
            runtime._run_client_round(
                {"job_id": "harbour-fl-demo", "coordinator_id": "coordinator", "filename": "model"}, [], [], [], Mock()
            )


@unittest.skipUnless(
    importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
    "Install ONNX and the detection extra for the real runtime fixture",
)
class OnnxRuntimeTests(unittest.TestCase):
    def test_weights_changed_after_configuration_require_worker_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_config(configuration(root))
            config.model.write_bytes(b"changed-model")
            with self.assertRaisesRegex(ValueError, "restart the worker"):
                YoloDetector(config)

    def test_real_session_letterboxes_decodes_and_crops_a_small_fixture(self):
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.array([16, 16, 16, 8, 0.9], dtype=np.float32).reshape(1, 5, 1)
            graph = helper.make_graph(
                [helper.make_node("Constant", [], ["output"], value=numpy_helper.from_array(values))],
                "fixture",
                [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])],
                [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 5, 1])],
            )
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
            model.ir_version = 10
            onnx.save(model, root / "model.onnx")
            config = load_config(configuration(root, cache=False))
            image = Image.new("RGB", (64, 32), "blue")
            ImageDraw.Draw(image).rectangle((16, 8, 47, 23), fill="red")
            crop, result = CameraPreprocessor(config, YoloDetector(config)).process(image)
            self.assertEqual(result["box"], [16, 8, 48, 24])
            self.assertEqual(crop.size, (32, 16))
            self.assertEqual(crop.getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(hashlib.sha256((root / "model.onnx").read_bytes()).hexdigest(), config.model_sha256)
