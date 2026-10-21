"""Load application-owned scenarios and derive consistent browser defaults."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def load_catalog(path: Path) -> dict[str, Any]:
    """Load inline or project-local scenario definitions, rejecting ambiguous or escaping file references."""
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if "scenarios_file" in catalog:
        source = (path.parent / catalog["scenarios_file"]).resolve()
        if not source.is_relative_to(path.parent.resolve()):
            raise ValueError("scenario file escapes its project")
        if "scenarios" in catalog:
            raise ValueError("use scenarios_file or inline scenarios, not both")
        catalog["scenarios"] = json.loads(source.read_text(encoding="utf-8"))
    return catalog


def parameter_definitions(catalog: dict[str, Any], name: str) -> dict[str, Any]:
    """Merge parameter schemas and scenario defaults without modifying the source catalog."""
    scenario = catalog["scenarios"][name]
    definitions = copy.deepcopy({**catalog.get("parameters", {}), **scenario.get("parameters", {})})
    for key, definition in definitions.items():
        if key in scenario:
            definition["default"] = scenario[key]
    return definitions


def effective_scenario(catalog: dict[str, Any], name: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Copy a scenario and resolve its parameters using operator values ahead of declared defaults."""
    scenario = copy.deepcopy(catalog["scenarios"][name])
    for key, definition in parameter_definitions(catalog, name).items():
        scenario[key] = parameters.get(key, definition.get("default"))
    return scenario
