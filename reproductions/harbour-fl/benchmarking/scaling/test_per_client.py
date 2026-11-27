"""Validate fixed client workload, explicit reuse, isolated holdouts and large manifest projection."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cluster import dataset_configmaps
from common import ROOT, load
from measurements import curves, summarise
from partitions import prepare, rows
from runner import validate
from test_scaling import result_inputs, snapshot


class PerClientTests(unittest.TestCase):
    def settings(self):
        return load(ROOT / "settings-per-client.json")

    def test_fixed_workload_grows_with_clients_and_client_assignments_remain_stable(self):
        settings = self.settings()
        validate(settings)
        previous = None
        for count in (1, 3, 6, 12, 24):
            data, report = prepare(snapshot(), count, 17, settings)
            self.assertEqual(report["samples"], {"train": 160 * count, "validation": 32 * count, "test": 10})
            self.assertEqual(report["pool_samples"], {"train": 37, "validation": 12, "test": 10})
            self.assertEqual(len(rows(data["clients/vessel_1/train_manifest.csv"])), 160)
            self.assertEqual(report["reuse"]["train"]["reused_occurrences"], count * 160 - 37)
            for client, splits in (previous or {}).items():
                self.assertEqual(report["clients"][client], splits)
            previous = report["clients"]

    def test_reuse_never_crosses_source_split_boundaries(self):
        data, report = prepare(snapshot(), 24, 17, self.settings())
        train = {key for client in report["clients"].values() for key in client["train"]["ids"]}
        val = {key for client in report["clients"].values() for key in client["val"]["ids"]}
        test = {r["segment_id"] for r in rows(data["test_manifest.csv"])}
        self.assertFalse(train & val or train & test or val & test)
        self.assertEqual(report["source_split_hashes"]["test"], report["split_hashes"]["test"])

    def test_no_repeats_within_client_when_quota_fits_source_pool(self):
        settings = {**self.settings(), "train_samples_per_client": 20, "validation_samples_per_client": 8}
        _, report = prepare(snapshot(), 24, 17, settings)
        for client in report["clients"].values():
            self.assertEqual(client["train"]["unique_samples"], 20)
            self.assertEqual(client["val"]["unique_samples"], 8)

    def test_existing_fixed_total_partitions_are_unchanged(self):
        old = prepare(snapshot(), 12, 17)
        new = prepare(snapshot(), 12, 17, load(ROOT / "settings.json"))
        self.assertEqual(old, new)

    def test_large_dataset_views_are_split_without_losing_or_duplicate_paths(self):
        files = {f"clients/vessel_{i}/train_manifest.csv": "x" * 60000 for i in range(24)}
        maps, volume = dataset_configmaps("et-benchmark-scaling-per-client", files)
        self.assertGreater(len(maps), 1)
        by_name = {doc["metadata"]["name"]: doc for doc in maps}
        restored = {}
        for source in volume["projected"]["sources"]:
            config = source["configMap"]
            doc = by_name[config["name"]]
            self.assertLessEqual(sum(len(v.encode()) for v in doc["data"].values()), 800000)
            for item in config["items"]:
                self.assertNotIn(item["path"], restored)
                restored[item["path"]] = doc["data"][item["key"]]
        self.assertEqual(restored, files)

    def test_curve_labels_and_processed_work_describe_per_client_study(self):
        settings, partition, logs, samples = result_inputs()
        report = {
            "study_design": "fixed_per_client",
            "clients": 1,
            "repetition": 1,
            "summary": summarise(logs, samples, partition, settings),
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            curves(root, {"c01-r01": report})
            self.assertIn("reused samples", (root / "scaling.svg").read_text())
            self.assertEqual(rows((root / "scaling.csv").read_text())[0]["study_design"], "fixed_per_client")
            other = {**report, "study_design": "fixed_total"}
            with self.assertRaisesRegex(ValueError, "separate curves"):
                curves(root, {"c01-r01": report, "c03-r01": other})

    def test_invalid_client_workload_is_rejected(self):
        settings = self.settings()
        settings["train_samples_per_client"] = 0
        with self.assertRaises(ValueError):
            validate(settings)

    def test_cli_plan_defaults_to_fixed_per_client_settings(self):
        env = {**os.environ, "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run(
            [sys.executable, str(ROOT / "cli.py"), "plan"], env=env, text=True, capture_output=True, check=True
        )
        plan = json.loads(result.stdout)
        self.assertEqual(plan["settings"]["study_design"], "fixed_per_client")
        self.assertEqual(plan["settings"]["train_samples_per_client"], 160)


if __name__ == "__main__":
    unittest.main()
