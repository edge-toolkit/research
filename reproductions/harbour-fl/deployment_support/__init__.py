"""Shared deployment settings; not part of the training runtime."""

import os
from pathlib import Path

ROOT_MODULE = "@edge-toolkit/et-ws-server-static"
RUNNER_IMAGE = "harbour-fl-pyo3:local"
DATASET_PATH = "/var/lib/edge-toolkit/harbour-fl-data"
PROJECT = Path(__file__).resolve().parents[1]


def detector_environment(*, container=False):
    """Build native or container detector settings, validating enabled models against the supported paths."""
    enabled = os.environ.get("HARBOUR_FL_DETECTOR_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    default = PROJECT / "models/vessel-detector/detector.json"
    configured = Path(os.environ.get("HARBOUR_FL_DETECTOR_CONFIG", default)).expanduser()
    path = (PROJECT / configured).resolve()
    if enabled:
        from harbour_fl.vision.config import load_config

        settings = load_config(path)
        if container and (
            path != default.resolve() or not settings.model.is_relative_to((PROJECT / "models").resolve())
        ):
            raise ValueError(
                "container detection requires models/vessel-detector/detector.json and weights under models/"
            )
    return {
        "HARBOUR_FL_DETECTOR_ENABLED": str(enabled).lower(),
        "HARBOUR_FL_DETECTOR_CONFIG": "/opt/models/vessel-detector/detector.json" if container else str(path),
        "HARBOUR_FL_DETECTOR_CACHE": "/tmp/harbour-fl-detector-cache"
        if container
        else str(PROJECT / "results/detector-cache"),
    }
