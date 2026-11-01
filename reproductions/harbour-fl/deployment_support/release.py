"""Harbour-FL deployment support: release."""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def prepare_release(path: Path) -> None:
    """Stage matching released Python wheels, tooling and browser modules; never fall back to an old release."""
    required = [
        path / "bin/et-cli",
        path / "images/core-runtime.tar",
        path / "modules/@edge-toolkit/et-ws-pyo3-mlflow/package.json",
        path / "modules/@edge-toolkit/et-ws-server-static/package.json",
        path / "modules/@edge-toolkit/et-ws-wasm-agent/package.json",
    ]
    wheels = list((path / "python").glob("*.whl"))
    missing = [str(item.relative_to(path)) for item in required if not item.is_file()]
    for package in ("edge_toolkit", "et_ws"):
        if not any(wheel.name.startswith(package + "-") for wheel in wheels):
            missing.append(f"python/{package}-*.whl")
    if missing:
        raise RuntimeError(
            "The configured local Core release does not include recovery supervision. "
            "Rebuild that release first; missing: " + ", ".join(missing)
        )
    project = PROJECT_ROOT
    destination = project / ".local-core-python"
    destination.mkdir(exist_ok=True)
    expected_names = {wheel.name for wheel in wheels}
    for stale in destination.glob("*.whl"):
        if stale.name not in expected_names:
            stale.unlink()
    for wheel in wheels:
        shutil.copy2(wheel, destination / wheel.name)
    for package, folder in (
        ("et-ws-pyo3-mlflow", "pyo3-mlflow"),
        ("et-ws-server-static", "static"),
        ("et-ws-wasm-agent", "ws-wasm-agent"),
    ):
        shutil.copytree(
            path / f"modules/@edge-toolkit/{package}", project / f".local-modules/{folder}/pkg", dirs_exist_ok=True
        )
