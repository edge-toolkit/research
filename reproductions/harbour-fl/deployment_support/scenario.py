"""Harbour-FL deployment support: scenario."""

from __future__ import annotations

import json
import os
import site
from pathlib import Path

from . import ROOT_MODULE, detector_environment


def configure_scenario(path: Path) -> Path:
    """Keep one application scenario; let Core generate optional recovery resources for k3s only."""
    import yaml

    scenario = yaml.safe_load(path.read_text())
    recovery = json.loads(path.with_name("recovery.json").read_text())
    scenario["deployment_type"] = "k3s"
    scenario["recovery"] = {
        key: recovery[key] for key in ("job_id", "mode", "poll_seconds", "command_timeout_seconds", "policy")
    }
    scenario["recovery"]["workloads"] = [
        {"deployment": name, "agent_id": name, "source": "/modules/@edge-toolkit/harbour-fl-engine"}
        for name in recovery["workloads"]
    ]
    if "resilience" in recovery:
        scenario["recovery"]["resilience"] = recovery["resilience"]
    for agent in scenario["agents"]:
        if agent["name"] in recovery["workloads"]:
            agent.setdefault("env", {})["HARBOUR_FL_JOB_ID"] = recovery["job_id"]
    output = path.with_name(".scenario-k3s.yaml")
    output.write_text(yaml.safe_dump(scenario, sort_keys=False))
    return output


def configure_mise(path: Path) -> None:
    """Set the staged Core frontend as the generated mise deployment's root."""
    text = path.read_text(encoding="utf-8")
    old = 'MODULES_ROOT = ""'
    if text.count(old) != 1:
        raise RuntimeError(f"expected one empty MODULES_ROOT in {path}")
    if text.count("[env]\n") != 1:
        raise RuntimeError(f"expected one global environment table in {path}")
    settings = "\n".join(f"{key} = {json.dumps(value)}" for key, value in detector_environment().items())
    text = text.replace("[env]\n", "[env]\n" + settings + "\n", 1)
    native_path = os.pathsep.join(["../src", *site.getsitepackages(), "/opt/et-python"])
    text = text.replace('"../src:/opt/et-python"', json.dumps(native_path))
    path.write_text(text.replace(old, f'MODULES_ROOT = "{ROOT_MODULE}"'), encoding="utf-8")
