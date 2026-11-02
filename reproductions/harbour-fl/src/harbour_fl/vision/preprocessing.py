"""Camera cropping with content-addressed box caching and explicit missing-detection policies."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
import uuid
from functools import lru_cache
from pathlib import Path

from .config import PROJECT_ROOT, detector_config
from .yolo import YoloDetector

logger = logging.getLogger(__name__)


class CameraPreprocessor:
    def __init__(self, config, detector=None):
        self.config = config
        self.detector = detector
        self.cache_dir = Path(os.environ.get("HARBOUR_FL_DETECTOR_CACHE", PROJECT_ROOT / "results/detector-cache"))
        self.cache_writable = True
        self.counts = dict.fromkeys(
            ("images", "crops", "full_image_fallbacks", "zero_fallbacks", "cache_hits", "inferences"), 0
        )
        self.seconds = 0.0

    def valid_prediction(self, result, size):
        """Validate cached or inferred vessel boxes, confidence scores and selected class IDs against image bounds."""
        if not isinstance(result, dict) or not {"box", "confidence", "class_id"}.issubset(result):
            return False
        if result.get("box") is None:
            return result.get("confidence") is None and result.get("class_id") is None
        box, score = result["box"], result.get("confidence")
        return (
            isinstance(box, list)
            and len(box) == 4
            and all(type(x) is int for x in box)
            and 0 <= box[0] < box[2] <= size[0]
            and 0 <= box[1] < box[3] <= size[1]
            and type(score) in (int, float)
            and math.isfinite(score)
            and self.config.confidence <= score <= 1
            and type(result.get("class_id")) is int
            and result["class_id"] in self.config.class_ids
        )

    def cached(self, path, size):
        """Reuse only cache records matching the image geometry, detector signature and prediction schema."""
        try:
            value = json.loads(path.read_text())
            if (
                value.get("schema_version") == 1
                and value.get("signature") == self.config.fingerprint()
                and value.get("size") == list(size)
                and self.valid_prediction(value.get("prediction"), size)
            ):
                return value["prediction"]
        except (OSError, ValueError, AttributeError):
            return None  # Missing, stale or corrupt cache entries are recomputed from the source image.
        return None

    def store(self, path, prediction, size):
        """Atomically cache a prediction, falling back to uncached inference when the cache cannot be written."""
        if not self.cache_writable:
            return
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            temporary.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "signature": self.config.fingerprint(),
                        "size": list(size),
                        "prediction": prediction,
                    }
                )
            )
            temporary.replace(path)
        except OSError:
            self.cache_writable = False
            logger.warning("Detector cache is unwritable; continuing with uncached detector inference")
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                logger.debug("Unable to remove detector cache temporary file", exc_info=True)

    def process(self, image):
        """Select or reuse a vessel crop and apply the configured no-detection policy while tracking work."""
        from PIL import Image

        started = time.monotonic()
        self.counts["images"] += 1
        path = None
        prediction = None
        if self.config.cache:
            identity = hashlib.sha256(image.tobytes()).hexdigest()
            key = hashlib.sha256(f"{self.config.fingerprint()}:{image.size}:{identity}".encode()).hexdigest()
            path = self.cache_dir / self.config.model_sha256 / f"{key}.json"
            prediction = self.cached(path, image.size)
        hit = prediction is not None
        if hit:
            self.counts["cache_hits"] += 1
        else:
            if self.detector is None:
                self.detector = YoloDetector(self.config)
            prediction = self.detector.predict(image)
            if not self.valid_prediction(prediction, image.size):
                raise ValueError("detector returned an invalid crop")
            self.counts["inferences"] += 1
            if path is not None:
                self.store(path, prediction, image.size)
        if prediction["box"] is not None:
            result = image.crop(tuple(prediction["box"]))
            self.counts["crops"] += 1
            outcome = "cropped"
        elif self.config.no_detection == "full-image":
            result = image
            self.counts["full_image_fallbacks"] += 1
            outcome = "full-image"
        elif self.config.no_detection == "zero":
            result = Image.new("RGB", image.size)
            self.counts["zero_fallbacks"] += 1
            outcome = "zero"
        else:
            raise ValueError("no vessel detected and no_detection=error")
        self.seconds += time.monotonic() - started
        return result, {**prediction, "outcome": outcome, "cache_hit": hit, "preprocessing": self.config.metadata()}


@lru_cache(maxsize=4)
def processor(config):
    return CameraPreprocessor(config)


def crop_camera(image):
    """Apply configured vessel cropping or return the complete image when detection is disabled."""
    config = detector_config()
    return processor(config).process(image) if config is not None else (image, {"outcome": "disabled"})


def detection_counters():
    """Return cumulative detector counts and elapsed inference time, or no counters when disabled."""
    config = detector_config()
    if config is None:
        return {}
    instance = processor(config)
    return {**instance.counts, "seconds": instance.seconds}


def detection_delta(before):
    """Report per-operation changes from a previous snapshot of detector counters."""
    return {f"camera_detection_{key}": float(value - before.get(key, 0)) for key, value in detection_counters().items()}
