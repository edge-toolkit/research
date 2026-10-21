"""Retain failed assessment evidence and produce strict JSON across report-writing paths."""

import contextlib
import csv
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import Mock, patch

from et_experiments.benchmarking.engine import BenchmarkEngine
from et_experiments.benchmarking.reports import save_report
from et_experiments.benchmarking.series import execute_series
from et_experiments.serialization import finite_number, json_dumps, json_safe
from et_experiments.utils import BenchmarkError
from test_engine import ImageOnlyAdapter


def strict_load(path):
    """Reject the nonstandard NaN and Infinity tokens accepted by Python's default decoder."""

    def reject(value):
        raise AssertionError(f"Nonstandard JSON token: {value}")

    return json.loads(path.read_text(), parse_constant=reject)


class SerializationTests(unittest.TestCase):
    def test_recursive_sanitizer_preserves_values_without_mutating_input(self):
        value = {"metrics": [float("nan"), float("inf"), -float("inf"), 0, False, "NaN"], "tuple": (1, None)}
        expected = {"metrics": [None, None, None, 0, False, "NaN"], "tuple": [1, None]}
        self.assertEqual(json_safe(value), expected)
        self.assertEqual(json.loads(json_dumps(value)), expected)
        self.assertTrue(math.isnan(value["metrics"][0]))
        for invalid in (True, False, None, float("nan"), float("inf")):
            self.assertFalse(finite_number(invalid))
        for valid in (0, -2, 1.5):
            self.assertTrue(finite_number(valid))

    def test_detailed_and_public_reports_write_nonfinite_metrics_as_null(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            public = output / "public.json"
            report = {
                "job_id": "example",
                "status": "passed",
                "sli": {"loss": float("nan")},
                "samples": [{"memory": float("inf")}],
                "resilience": {"value": -float("inf")},
            }
            with patch.dict(os.environ, {"BENCHMARK_RESULT_FILE": str(public)}):
                save_report(output, report)
            detailed = strict_load(output / "example.json")
            self.assertIsNone(detailed["sli"]["loss"])
            self.assertIsNone(detailed["samples"][0]["memory"])
            self.assertIsNone(strict_load(public)["resilience"]["value"])

    def test_series_and_browser_summary_files_use_strict_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            engine = Mock()
            engine.execute.return_value = {
                "job_id": "example",
                "scenario": "example",
                "status": "passed",
                "sli": {"loss": float("nan")},
            }
            public = output / "public.json"
            with patch.dict(os.environ, {"BENCHMARK_RESULT_FILE": str(public)}):
                document = execute_series(engine, "example", {}, output)
            self.assertIsNone(strict_load(output / f"{document['series_id']}.json")["runs"][0]["sli"]["loss"])
            self.assertIsNone(strict_load(public)["run_reports"][0]["sli"]["loss"])


class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.app = ImageOnlyAdapter()
        self.backend = Mock()
        self.backend.kubectl.return_value = CompletedProcess([], 0, stdout=self.app.config.context)
        self.backend.primary_logs.return_value = "completed"
        self.backend.collect_logs.return_value = {}
        self.backend.warning_events.return_value = [{"reason": "example"}]
        self.engine = BenchmarkEngine(self.app, backend=self.backend)
        self.engine.sampler_factory = Mock(return_value=Mock(samples=[], metrics_available=False))
        self.engine.faults.remove_network_fault_sidecar = Mock()

    def test_assessment_failures_never_persist_passed_status_or_discard_measurements(self):
        targets = ("adapter", "recovery_evidence", "evaluate_slos")
        for target in targets:
            for kind in (KeyError, TypeError, BenchmarkError):
                with self.subTest(target=target, kind=kind.__name__), tempfile.TemporaryDirectory() as directory:
                    error = kind("assessment failed")
                    hook = (
                        patch.object(self.app, "enrich_report", side_effect=error)
                        if target == "adapter"
                        else patch(f"et_experiments.benchmarking.engine.{target}", side_effect=error)
                    )
                    with hook, contextlib.redirect_stdout(io.StringIO()), self.assertRaises(kind) as raised:
                        self.engine.execute(
                            "assessment", {"batches": 1, "recovery_enabled": False}, Path(directory), 2, False
                        )
                    self.assertIs(raised.exception, error)
                    reports = [strict_load(path) for path in Path(directory).glob("*.json")]
                    report = next(value for value in reports if value.get("scenario") == "assessment")
                    self.assertEqual(report["status"], "failed")
                    self.assertEqual(report["workload"]["completed_units"], 1)
                    self.assertEqual(report["warning_events"], [{"reason": "example"}])
                    self.assertEqual(report["cleanup"]["status"], "restored")
                    with (Path(directory) / "summary.csv").open() as source:
                        self.assertEqual(next(csv.DictReader(source))["status"], "failed")

    def test_engine_report_and_result_paths_sanitize_optional_metrics(self):
        enrich = self.app.enrich_report

        def add_measurements(report):
            report["resilience"]["invalid"] = float("nan")
            return {**enrich(report), "invalid": float("inf")}

        self.app.enrich_report = add_measurements
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with (
                patch.dict(os.environ, {"BENCHMARK_RESULT_FILE": str(output / "public.json")}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                report = self.engine.execute("finite", {"batches": 1, "recovery_enabled": False}, output, 2, False)
            self.assertEqual(report["status"], "passed")
            detailed = strict_load(output / f"{report['job_id']}.json")
            self.assertIsNone(detailed["sli"]["invalid"])
            self.assertIsNone(strict_load(output / "public.json")["resilience"]["invalid"])
