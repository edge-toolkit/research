"""Validate 24-client partitions, resource admission and server-side apply."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from capacity import check_capacity
from cli import arguments
from cluster import Cluster, documents
from common import ROOT, load
from partitions import csv_text, prepare, rows
from runner import jobs, validate
from test_scaling import snapshot


def expanded_snapshot():
    """Provide enough distinct validation samples for 24 nonempty client shards."""
    value = snapshot()
    for name in list(value["manifests"]):
        if name.endswith("/val_manifest.csv"):
            original = rows(value["manifests"][name])
            extra = [{**r, "segment_id": r["segment_id"] + "extra"} for r in original]
            value["manifests"][name] = csv_text(original + extra, tuple(original[0]))
            value["usable_ids"][name] += [r["segment_id"] for r in extra]
            for r in extra:
                value["manifests"]["feature_manifest.csv"] += r["segment_id"] + ",feature.npy\n"
    return value


class ExtensionTests(unittest.TestCase):
    def test_empty_path_options_are_rejected_before_launch(self):
        for option in ("--settings=", "--output="):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "requires a path"):
                arguments(["run", option])

    def test_24_client_point_keeps_existing_splits_and_resource_policy(self):
        value = expanded_snapshot()
        settings = load(ROOT / "settings.json")
        validate(settings)
        self.assertEqual(jobs(settings)[-1], ("c24-r01", 24, 1))
        _, previous = prepare(value, 12, settings["seed"])
        data, partition = prepare(value, 24, settings["seed"])
        self.assertEqual(previous["split_hashes"], partition["split_hashes"])
        self.assertEqual(partition["samples"], previous["samples"])
        self.assertEqual(len(partition["clients"]), 24)
        docs = documents(settings, value, data, 24, "test")
        roles = [doc for doc in docs if doc["kind"] == "Deployment"]
        self.assertEqual(len(roles), 26)
        last = next(doc for doc in roles if doc["metadata"]["name"] == "vessel-24")
        container = last["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(container["resources"], settings["resources"]["client"])
        env = {e["name"]: e["value"] for e in container["env"]}
        self.assertEqual(env["HARBOUR_FL_CLIENT_ID"], "vessel_24")
        self.assertEqual(len(env["HARBOUR_FL_CLIENTS"].split(",")), 24)

    def test_output_path_is_resolved_before_mise_changes_directory(self):
        result = arguments(["run", "--output", "results/current", "--case", "24"])
        self.assertTrue(Path(result[2]).is_absolute())

    def test_24_client_memory_overcommit_requires_verified_reference(self):
        class FakeCluster:
            def get(self, *_args):
                return {"items": []}

            def call(self, *_args):
                return json.dumps({"usage": {"memory": "8Gi"}})

        settings = load(ROOT / "settings.json")
        snap = {"node": "worker", "allocatable": {"cpu": "20", "memory": "120Gi"}}
        with self.assertRaisesRegex(RuntimeError, "12-client point"):
            check_capacity(FakeCluster(), settings, snap, 24, {})
        roles = {
            "ws-server": {"peak_memory_mib": 200},
            "vessel-coordinator": {"peak_memory_mib": 1200},
            **{f"vessel-{i + 1}": {"peak_memory_mib": 1200} for i in range(12)},
        }
        reports = {
            "c12-r01": {
                "clients": 12,
                "status": "passed",
                "cleanup_verified": True,
                "summary": {"resources": {"roles": roles}},
            }
        }
        result = check_capacity(FakeCluster(), settings, snap, 24, reports)
        self.assertEqual(result["study_requests"]["cpu"], 12.6)
        self.assertEqual(result["reference_jobs"], ["c12-r01"])
        snap["allocatable"]["cpu"] = "12"
        with self.assertRaisesRegex(RuntimeError, "cpu requests"):
            check_capacity(FakeCluster(), settings, snap, 24, reports)


class ServerApplyTests(unittest.TestCase):
    def test_large_configmap_uses_server_side_apply_without_forcing_conflicts(self):
        cluster = object.__new__(Cluster)
        cluster.namespace = "et-benchmark-scaling-per-client"
        cluster.call = Mock(side_effect=[json.dumps({"metadata": {"uid": "created"}}), "applied"])
        document = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "scaling-data", "namespace": cluster.namespace},
            "data": {"manifest.csv": "x" * 300000},
        }
        with tempfile.TemporaryDirectory() as folder:
            cluster.namespace_journal = Path(folder) / "namespace-owner.json"
            namespace = {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": cluster.namespace}}
            cluster.apply([namespace, document])
        args, kwargs = cluster.call.call_args
        self.assertIn("--server-side", args)
        self.assertIn("--field-manager=harbour-fl-scaling", args)
        self.assertNotIn("--force-conflicts", args)
        payload = json.loads(kwargs["stdin"])
        self.assertEqual(payload["items"], [document])
        self.assertNotIn("annotations", payload["items"][0]["metadata"])

    def test_namespace_boundary_is_checked_before_applying(self):
        cluster = object.__new__(Cluster)
        cluster.namespace = "et-benchmark-scaling-per-client"
        cluster.call = Mock()
        with self.assertRaises(ValueError):
            cluster.apply([{"kind": "ConfigMap", "metadata": {"namespace": "et-harbour-fl"}}])
        cluster.call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
