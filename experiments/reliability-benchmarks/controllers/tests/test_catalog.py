"""Scenario defaults must agree between the CLI, adapter, and browser."""

import json
import tempfile
import unittest
from pathlib import Path

from et_experiments.catalog import effective_scenario, load_catalog
from et_experiments.web.controller import BenchmarkController


class CatalogTests(unittest.TestCase):
    def test_scenario_file_owns_descriptions_and_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenarios = {"restore": {"description": "Restore state", "rounds": 5, "timeout_seconds": 300}}
            (root / "scenarios.json").write_text(json.dumps(scenarios))
            catalog = {
                "schema_version": 1,
                "adapter": {"command": ["python", "adapter.py"]},
                "scenarios_file": "scenarios.json",
                "parameters": {
                    "rounds": {"type": "integer", "default": 3},
                    "timeout_seconds": {"type": "integer", "default": 900},
                },
            }
            path = root / "catalog.json"
            path.write_text(json.dumps(catalog))
            controller = BenchmarkController(path)
            name, parameters = controller.validate_request({"scenario": "restore"})
            self.assertEqual(parameters, {"rounds": 5, "timeout_seconds": 300})
            public = controller.public_catalog()["scenarios"]["restore"]
            self.assertEqual(public["description"], "Restore state")
            self.assertEqual(public["parameters"]["rounds"]["default"], 5)
            effective = effective_scenario(load_catalog(path), name, {"rounds": 8})
            self.assertEqual(effective["rounds"], 8)
            self.assertEqual(effective["timeout_seconds"], 300)
            self.assertEqual(load_catalog(path)["scenarios"]["restore"]["rounds"], 5)

    def test_scenario_file_cannot_escape_project(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(json.dumps({"scenarios_file": "../outside.json"}))
            with self.assertRaisesRegex(ValueError, "escapes its project"):
                load_catalog(path)
