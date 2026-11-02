"""Frozen YOLOv8 ONNX inference, letterboxing, vessel filtering and crop selection."""

from __future__ import annotations

import hashlib
import math


def overlap(first, second):
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    total = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(total - intersection, 1e-12)


def select_crop(predictions, config, image_size, scale, padding):
    """Decode raw [1, 4+classes, anchors] predictions in letterboxed input coordinates."""
    import numpy as np

    output = np.asarray(predictions)
    if (
        output.ndim != 3
        or output.shape[0] != 1
        or output.shape[1] != 4 + config.num_classes
        or not 1 <= output.shape[2] <= 100000
        or not np.isfinite(output).all()
    ):
        raise ValueError("expected finite YOLOv8 raw output [1, 4+num_classes, anchors]; export with nms=False")
    rows = output[0].T
    scores = rows[:, 4:]
    if (scores < 0).any() or (scores > 1).any():
        raise ValueError("YOLOv8 class scores must be probabilities between 0 and 1")
    classes = scores.argmax(axis=1)
    confidences = scores.max(axis=1)
    indices = np.flatnonzero(np.isin(classes, config.class_ids) & (confidences >= config.confidence))
    indices = sorted(indices, key=lambda i: (-float(confidences[i]), int(i)))[:1000]
    width, height = image_size
    sx, sy = scale
    px, py = padding
    candidates = []
    for index in indices:
        x, y, w, h = (float(value) for value in rows[index, :4])
        if w <= 0 or h <= 0:
            continue
        box = (
            max(0.0, (x - w / 2 - px) / sx),
            max(0.0, (y - h / 2 - py) / sy),
            min(float(width), (x + w / 2 - px) / sx),
            min(float(height), (y + h / 2 - py) / sy),
        )
        if box[2] > box[0] and box[3] > box[1]:
            candidates.append({"box": box, "confidence": float(confidences[index]), "class_id": int(classes[index])})
    kept = []
    for candidate in candidates:
        if all(overlap(candidate["box"], other["box"]) <= config.iou for other in kept):
            kept.append(candidate)
        if len(kept) >= 100:
            break
    if not kept:
        return {"box": None, "confidence": None, "class_id": None}

    def ranking(candidate):
        """Rank boxes by the configured primary criterion with deterministic secondary ordering."""
        box = candidate["box"]
        area = (box[2] - box[0]) * (box[3] - box[1])
        return (area, candidate["confidence"]) if config.selection == "largest" else (candidate["confidence"], area)

    selected = max(kept, key=ranking)
    left, top, right, bottom = selected["box"]
    mx, my = (right - left) * config.margin, (bottom - top) * config.margin
    selected["box"] = [
        max(0, math.floor(left - mx)),
        max(0, math.floor(top - my)),
        min(width, math.ceil(right + mx)),
        min(height, math.ceil(bottom + my)),
    ]
    return selected


class YoloDetector:
    def __init__(self, config):
        """Verify ONNX weights and validate the supported CPU session input and output geometry."""
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "camera detection requires the detection extra: pip install -e '.[detection]'"
            ) from error
        options = ort.SessionOptions()
        with config.model.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != config.model_sha256:
                raise ValueError("detector weights changed after configuration was loaded; restart the worker")
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(config.model), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)" or inputs[0].shape != [1, 3, config.size, config.size]:
            raise ValueError("export a static batch=1, FP32 YOLOv8 ONNX model with matching input_size")
        if (
            len(outputs) != 1
            or outputs[0].type != "tensor(float)"
            or len(outputs[0].shape) != 3
            or outputs[0].shape[:2] != [1, 4 + config.num_classes]
        ):
            raise ValueError("export a detection-only YOLOv8 model with nms=False and matching num_classes")
        self.config = config
        self.input_name = inputs[0].name

    def predict(self, image):
        import numpy as np
        from PIL import Image

        width, height = image.size
        ratio = min(self.config.size / width, self.config.size / height)
        resized = (max(1, round(width * ratio)), max(1, round(height * ratio)))
        padding = ((self.config.size - resized[0]) // 2, (self.config.size - resized[1]) // 2)
        canvas = Image.new("RGB", (self.config.size, self.config.size), (114, 114, 114))
        canvas.paste(image.resize(resized, Image.Resampling.BILINEAR), padding)
        inputs = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        output = self.session.run(None, {self.input_name: np.ascontiguousarray(inputs)})[0]
        return select_crop(output, self.config, image.size, (resized[0] / width, resized[1] / height), padding)
