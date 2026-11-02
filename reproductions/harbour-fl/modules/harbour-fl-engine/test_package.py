"""The runner loads a thin application package without the external benchmarking tools."""

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

from harbour_fl.runtime import VesselRuntime

PROJECT = Path(__file__).resolve().parents[2]
ENTRY = Path(__file__).parent / "pkg/et_ws_pyo3_harbour_fl_demo.py"


class PackageTests(unittest.TestCase):
    def test_runner_entry_exports_the_same_hooks_without_benchmark_imports(self):
        script = """
import importlib.abc
import importlib.util
import sys

class NoBenchmark(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, _path, _target=None):
        if fullname == 'et_experiments' or fullname.startswith('et_experiments.'):
            raise AssertionError('Application runtime imported benchmark tooling')

sys.meta_path.insert(0, NoBenchmark())
spec = importlib.util.spec_from_file_location('runner_entry', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert all(callable(getattr(module, name)) for name in ('init', 'on_connect', 'on_text_frame', 'on_shutdown'))
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(ENTRY)], capture_output=True, text=True, timeout=10, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_runtime_instances_do_not_share_state_or_synchronization(self):
        first, second = VesselRuntime(), VesselRuntime()
        first.state["seen_rounds"].add(1)
        first.state["stop"].set()
        self.assertEqual(second.state["seen_rounds"], set())
        self.assertFalse(second.state["stop"].is_set())
        self.assertIsNot(first.state["update_queue"], second.state["update_queue"])

    def test_image_and_application_requirements_exclude_controller(self):
        dockerfile = (PROJECT / "Dockerfile.pyo3").read_text()
        self.assertIn("COPY src/harbour_fl /opt/et-python/harbour_fl", dockerfile)
        self.assertNotIn("experiment-controllers", dockerfile)
        self.assertNotIn("et_experiments", dockerfile)
        self.assertNotIn("controllers", (PROJECT / "requirements.txt").read_text())

    def test_runner_can_resolve_package_from_explicit_source_path(self):
        spec = importlib.util.find_spec("harbour_fl")
        self.assertEqual(Path(spec.origin).parent, PROJECT / "src/harbour_fl")
