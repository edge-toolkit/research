"""Validate a local detector or preview its camera crop."""

import argparse
import json
from pathlib import Path

from ..settings import _bool_env
from .config import detector_config, load_config
from .preprocessing import CameraPreprocessor
from .yolo import YoloDetector


def main():
    """Check detector configuration or write a standalone image-crop preview and its metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "preview", "enabled"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/detection-preview"))
    args = parser.parse_args()
    if args.command == "enabled":
        print(int(_bool_env("HARBOUR_FL_DETECTOR_ENABLED")))
        return 0
    config = load_config(args.config.resolve()) if args.config else detector_config(True)
    detector = YoloDetector(config)
    if args.command == "check":
        print(json.dumps(config.metadata(), indent=2))
        return 0
    if args.image is None:
        parser.error("preview requires --image")
    from PIL import Image, ImageOps

    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    crop, metadata = CameraPreprocessor(config, detector).process(image)
    args.output.mkdir(parents=True, exist_ok=True)
    crop.save(args.output / "crop.png")
    (args.output / "detection.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
