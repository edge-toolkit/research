"""Exercise dataset fairness, deployment isolation, evidence checks and failure cleanup without a cluster."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cluster import Cluster, documents, verify_application
from common import ROOT, load, save, sha
from measurements import curves, quantities, resource_summary, summarise
from partitions import asset_path, balance, csv_text, prepare, rows
from runner import execute, verified


def snapshot():
    """Make a small uneven, rare-class dataset and source templates containing unrelated state."""
    fields = ("segment_id", "standard_label", "feature_path")
    all_rows = [
        {
            "segment_id": f"s{i}",
            "standard_label": "rare" if i == 0 else ("a" if i % 2 else "b"),
            "feature_path": f"features/s{i}.npz",
        }
        for i in range(59)
    ]
    manifests = {
        "class_summary.csv": "standard_label,segments\nrare,1\na,29\nb,29\n",
        "feature_manifest.csv": csv_text(
            [{"segment_id": r["segment_id"], "feature_path": r["feature_path"]} for r in all_rows],
            ("segment_id", "feature_path"),
        ),
        "test_manifest.csv": csv_text(all_rows[49:], fields),
    }
    for index in range(3):
        manifests[f"clients/vessel_{index + 1}/train_manifest.csv"] = csv_text(all_rows[index:37:3], fields)
        manifests[f"clients/vessel_{index + 1}/val_manifest.csv"] = csv_text(all_rows[37 + index : 49 : 3], fields)
    template = {
        "securityContext": {"fsGroup": 10001},
        "containers": [
            {
                "name": "original",
                "image": "old:local",
                "envFrom": [{"secretRef": {"name": "original-secret"}}],
                "env": [
                    {"name": k, "value": v}
                    for k, v in {
                        "RUNNER_MODULE": "et_ws_pyo3_harbour_fl_demo",
                        "MODULES_PATHS": "/app/modules",
                        "HARBOUR_FL_READINESS_FILE": "/tmp/ready",
                        "RECOVERY_JOB_ID": "original",
                        "WS_SERVER_URL": "ws://original-hub:8080/ws",
                    }.items()
                ],
                "resources": {"limits": {"nvidia.com/gpu": "1"}},
                "volumeMounts": [{"name": "old-data", "mountPath": "/data"}],
            }
        ],
        "volumes": [{"name": "old-data", "persistentVolumeClaim": {"claimName": "original-data"}}],
    }
    return {
        "preparation": {
            "files": {},
            "input": {},
            "labels": ["rare", "a", "b"],
            "class_weights": [1, 1, 1],
            "feature_dimension": 960,
        },
        "manifests": manifests,
        "usable_ids": {
            name: [r["segment_id"] for r in rows(text)]
            for name, text in manifests.items()
            if name.endswith("_manifest.csv") and name != "feature_manifest.csv"
        },
        "source_templates": {name: copy.deepcopy(template) for name in ("ws-server", "vessel-1", "vessel-coordinator")},
        "images": {
            name: {"image": "old:local", "image_id": "sha256:original"}
            for name in ("ws-server", "vessel-1", "vessel-coordinator")
        },
        "node": "worker",
        "node_hostname": "worker",
        "dataset_host_path": "/prepared-dataset",
        "storage_class": "local-path",
    }


def result_inputs(count=1):
    """Represent complete CPU contributions with the expected processed samples and resource usage."""
    settings = load(ROOT / "settings.json")
    settings["rounds"] = 1
    _, partition = prepare(snapshot(), count, 17)
    metrics = {
        "participating_clients": count,
        "round_work_seconds": 10.0,
        "global_test_accuracy": 0.5,
        "global_test_macro_f1": 0.4,
    }
    logs = {"vessel-coordinator": "coordinator completed round 0: " + repr(metrics) + '\n{"kind": "completed"}'}
    for i in range(count):
        client = f"vessel_{i + 1}"
        m = {
            "processed_samples": partition["clients"][client]["train"]["samples"] * settings["local_epochs"],
            "gpu_enabled": 0,
            "global_checkpoint_bytes": 100,
            "update_bytes": 100,
        }
        logs[f"vessel-{i + 1}"] = f"{client} completed round 0: {m!r}"
    names = ["ws-server", "vessel-coordinator", *(f"vessel-{i + 1}" for i in range(count))]
    sample = {
        "items": [
            {
                "metadata": {"name": name + "-abc-def"},
                "timestamp": "2026-11-12T00:00:00Z",
                "window": "10s",
                "containers": [{"usage": {"cpu": "500m", "memory": "100Mi"}}],
            }
            for name in names
        ]
    }
    return settings, partition, logs, [sample]


class PartitionTests(unittest.TestCase):
    def test_fixed_total_and_holdouts_with_no_duplicates_for_every_count(self):
        original = snapshot()
        split_hashes = None
        for count in (1, 3, 6, 12):
            data, report = prepare(original, count, 17)
            self.assertEqual(report["samples"], {"train": 37, "validation": 12, "test": 10})
            sizes = [c["train"]["samples"] for c in report["clients"].values()]
            self.assertLessEqual(max(sizes) - min(sizes), 1)
            ids = [i for c in report["clients"].values() for i in c["train"]["ids"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(ids), 37)
            split_hashes = split_hashes or report["split_hashes"]
            self.assertEqual(split_hashes, report["split_hashes"])
            self.assertEqual(len([p for p in data if p.endswith("/val_manifest.csv")]), count)

    def test_deterministic_seed_and_approximately_stratified_classes(self):
        records = [{"segment_id": str(i), "standard_label": str(i % 4)} for i in range(131)]
        a = balance(records, 12, 17)
        self.assertEqual(a, balance(records, 12, 17))
        self.assertNotEqual(a, balance(records, 12, 18))
        for label in ("0", "1", "2", "3"):
            counts = [sum(r["standard_label"] == label for r in shard) for shard in a]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_deployed_loader_exclusions_are_retained(self):
        value = snapshot()
        key = "clients/vessel_1/train_manifest.csv"
        excluded = value["usable_ids"][key].pop()
        _, report = prepare(value, 3, 17)
        self.assertEqual(report["samples"]["train"], 36)
        self.assertNotIn(excluded, [i for c in report["clients"].values() for i in c["train"]["ids"]])

    def test_split_leakage_fails_before_deployment(self):
        value = snapshot()
        key = "clients/vessel_1/train_manifest.csv"
        value["manifests"]["test_manifest.csv"] = value["manifests"][key]
        value["usable_ids"]["test_manifest.csv"] = value["usable_ids"][key]
        with self.assertRaisesRegex(ValueError, "leakage"):
            prepare(value, 3, 17)

    def test_paths_cannot_escape_the_dataset_mount(self):
        for invalid in ("", "/home/file.png", "../file.png", "images/../../file.png"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                asset_path(invalid)


class DeploymentTests(unittest.TestCase):
    def test_every_count_uses_independent_hub_cpu_limits_and_readonly_dataset(self):
        value = snapshot()
        original = copy.deepcopy(value)
        settings = load(ROOT / "settings.json")
        for count in (1, 3, 6, 12):
            data, _ = prepare(value, count, 17)
            docs = documents(settings, value, data, count, "job")
            deployments = [doc for doc in docs if doc["kind"] == "Deployment"]
            self.assertEqual(len(deployments), count + 2)
            for doc in deployments:
                self.assertEqual(doc["metadata"]["namespace"], "et-benchmark-scaling")
                spec = doc["spec"]["template"]["spec"]
                self.assertNotIn("nodeName", spec)  # Scheduler must bind WaitForFirstConsumer storage.
                c = spec["containers"][0]
                self.assertNotIn("envFrom", c)
                self.assertNotIn("nvidia.com/gpu", c["resources"]["limits"])
                env = {e["name"]: e["value"] for e in c["env"]}
                if doc["metadata"]["name"] == "ws-server":
                    self.assertEqual(env["HTTP_WORKERS"], "2")
                    self.assertIn(
                        {"name": "storage", "persistentVolumeClaim": {"claimName": "scaling-storage"}}, spec["volumes"]
                    )
                else:
                    self.assertEqual(env["HARBOUR_FL_DEVICE"], "cpu")
                    self.assertEqual(len(env["HARBOUR_FL_CLIENTS"].split(",")), count)
                    self.assertEqual(env["RECOVERY_SUPERVISOR_AGENT_ID"], "")
                    self.assertIn(
                        {"name": "source-data", "mountPath": "/scaling-source", "readOnly": True}, c["volumeMounts"]
                    )
            self.assertEqual(value, original)

    def test_image_application_is_verified_before_a_settings_only_overlay(self):
        expected = {"learning.py": "weights", "protocol.py": "protocol"}
        value = {
            "code_hashes": expected.copy(),
            "expected_code_hashes": expected,
            "application_root": "/opt/et-python/harbour_fl",
            "model_signature": {"model_profile": "pretrained"},
            "expected_model_signature": {"model_profile": "pretrained"},
        }
        templates = snapshot()["source_templates"]
        verify_application(value, templates)
        for invalid in ("modified", "missing", "extra", "model", "volume"):
            changed, deployed = copy.deepcopy(value), copy.deepcopy(templates)
            if invalid == "modified":
                changed["code_hashes"]["protocol.py"] = "different"
            elif invalid == "missing":
                del changed["code_hashes"]["learning.py"]
            elif invalid == "extra":
                changed["code_hashes"]["override.py"] = "extra"
            elif invalid == "model":
                changed["model_signature"] = {"model_profile": "different"}
            else:
                deployed["vessel-coordinator"]["containers"][0]["volumeMounts"].append(
                    {"name": "override", "mountPath": "/opt/et-python"}
                )
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                verify_application(changed, deployed)

    def test_unowned_namespace_cannot_be_deleted(self):
        cluster = object.__new__(Cluster)
        cluster.namespace = "et-benchmark-scaling"
        cluster.call = lambda *_args: json.dumps({"metadata": {"labels": {}}})
        with self.assertRaisesRegex(RuntimeError, "ownership"):
            cluster.cleanup()


class MeasurementTests(unittest.TestCase):
    def test_complete_round_has_correct_sample_and_payload_throughput(self):
        settings, partition, logs, samples = result_inputs(12)
        summary = summarise(logs, samples, partition, settings)
        row = summary["rounds"][0]
        self.assertEqual(row["training_examples"], 74)
        self.assertEqual(row["training_examples_per_second"], 7.4)
        self.assertEqual(row["client_model_payload_bytes"], 2400)
        self.assertEqual(row["accepted_updates_per_second"], 1.2)

    def test_missing_client_or_dropped_samples_cannot_be_success(self):
        for invalid in ("missing", "dropped", "gpu"):
            settings, partition, logs, samples = result_inputs(3)
            if invalid == "missing":
                logs.pop("vessel-3")
            elif invalid == "dropped":
                count = partition["clients"]["vessel_3"]["train"]["samples"] * settings["local_epochs"]
                original = logs["vessel-3"]
                logs["vessel-3"] = original.replace(
                    f"'processed_samples': {count}", f"'processed_samples': {count - 1}"
                )
                self.assertNotEqual(logs["vessel-3"], original)
            else:
                logs["vessel-3"] = logs["vessel-3"].replace("'gpu_enabled': 0", "'gpu_enabled': 1")
            error = (
                self.assertRaisesRegex(ValueError, "Processed sample count")
                if invalid == "dropped"
                else self.assertRaises(ValueError)
            )
            with self.subTest(invalid=invalid), error:
                summarise(logs, samples, partition, settings)

    def test_metrics_windows_are_deduplicated_and_role_coverage_is_required(self):
        _, _, _, samples = result_inputs()
        summary = resource_summary(samples * 3, ["ws-server", "vessel-1", "vessel-coordinator"])
        self.assertEqual(summary["roles"]["ws-server"]["windows"], 1)
        self.assertEqual(summary["aggregate_mean_cpu_cores"], 1.5)
        with self.assertRaisesRegex(ValueError, "Missing resource"):
            resource_summary(samples, ["vessel-12"])

    def test_kubernetes_quantity_scales(self):
        self.assertAlmostEqual(quantities("250000000n"), 0.25)
        self.assertEqual(quantities("1Gi", True), 1024)
        self.assertAlmostEqual(quantities("512M", True), 512000000 / 1048576)

    def test_curve_exports_separate_smoke_from_measured_points(self):
        settings, partition, logs, samples = result_inputs()
        report = {"clients": 1, "repetition": 1, "summary": summarise(logs, samples, partition, settings)}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            curves(root, {"smoke-c01-r01": report, "c01-r01": report})
            self.assertEqual(len(rows((root / "scaling.csv").read_text())), 1)
            self.assertIn("<svg", (root / "scaling.svg").read_text())


class EvidenceTests(unittest.TestCase):
    def test_successful_point_collects_rounds_metrics_logs_and_cleanup(self):
        settings, _partition, logs, samples = result_inputs()

        class SuccessfulCluster:
            def target(self):
                return {"cluster_uid": "cluster", "namespace": "study", "server": "server"}

            def __init__(self):
                self.cleanups = 0

            def cleanup(self):
                self.cleanups += 1

            def apply(self, _docs):
                return None

            def call(self, *_args):
                return "scaled"

            def logs(self, name, *, pod=None):
                if pod != name + "-abc-def":
                    raise AssertionError("Expected checked pod logs")
                return logs.get(name, "hub log")

            def resources(self, **_kwargs):
                return samples[0]

            def get(self, kind):
                if kind == "pods":
                    return {
                        "items": [
                            {
                                "metadata": {"labels": {"app": name}, "uid": name, "name": name + "-abc-def"},
                                "status": {"containerStatuses": [{"imageID": "sha256:original", "restartCount": 0}]},
                            }
                            for name in ("ws-server", "vessel-1", "vessel-coordinator")
                        ]
                    }
                return {"items": []}

        cluster = SuccessfulCluster()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "attempt-01"
            with patch("runner.wait_rollouts"), patch("runner.wait_clients"):
                report = execute(cluster, settings, snapshot(), folder, "c01-r01", 1, 1)
            self.assertEqual(report["status"], "passed")
            self.assertTrue(report["cleanup_verified"])
            self.assertEqual(cluster.cleanups, 2)
            self.assertEqual(len(rows((folder / "rounds.csv").read_text())), 1)
            self.assertEqual((folder / "vessel-1.log").read_text(), logs["vessel-1"])
            self.assertEqual(load(folder / "report.json")["summary"]["mean_examples_per_second"], 7.4)

    def test_resume_requires_intact_artifacts_and_verified_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            save(folder / "report.json", {"status": "passed"})
            save(
                folder / "completion.json",
                {
                    "signature": "same",
                    "cleanup_verified": True,
                    "artifacts": {"report.json": sha(folder / "report.json")},
                },
            )
            self.assertTrue(verified(folder, "same"))
            self.assertFalse(verified(folder, "different"))
            save(folder / "report.json", {"status": "changed"})
            self.assertFalse(verified(folder, "same"))

    def test_cleanup_failure_does_not_replace_original_apply_error(self):
        class BrokenCluster:
            def target(self):
                return {"cluster_uid": "cluster", "namespace": "study", "server": "server"}

            def __init__(self):
                self.cleanups = 0

            def cleanup(self):
                self.cleanups += 1
                if self.cleanups == 2:
                    raise RuntimeError("cleanup error")

            def apply(self, _docs):
                raise RuntimeError("original apply error")

            def logs(self, _name):
                return "retained log"

            def get(self, *_args):
                return {"items": []}

        cluster = BrokenCluster()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "attempt-01"
            with self.assertRaisesRegex(RuntimeError, "original apply error"):
                execute(cluster, load(ROOT / "settings.json"), snapshot(), folder, "test", 1, 1)
            report = load(folder / "report.json")
            self.assertFalse(report["cleanup_verified"])
            self.assertEqual(report["cleanup_error"], "cleanup error")
            self.assertIn("original apply error", report["error"])
            self.assertEqual(cluster.cleanups, 2)


if __name__ == "__main__":
    unittest.main()
