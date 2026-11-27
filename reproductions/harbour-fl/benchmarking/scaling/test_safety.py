"""Admission, pod continuity, cleanup identity and archive privacy regressions."""

import contextlib
import copy
import datetime as dt
import hashlib
import io
import itertools
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cli
import package
import runner
from capacity import request_totals
from cluster import (
    Cluster,
    authorized,
    dataset_host_path,
    documents,
    preflight,
    source_signature,
)
from common import CommandError, save, sources, tool_identity
from dataset_snapshot import copy_verified
from test_scaling import result_inputs, snapshot


class FixtureCluster:
    def __init__(self, *, replace_at=0, restart=False, interrupt=False, timeout=False):
        self.cleanups = 0
        self.pod_reads = 0
        self.log_reads = 0
        self.metric_reads = 0
        self.replace_at, self.restart, self.interrupt, self.timeout = replace_at, restart, interrupt, timeout
        self.settings, _, self.content, self.samples = result_inputs()

    def target(self):
        return {"server": "server", "cluster_uid": "cluster", "namespace": "study"}

    def cleanup(self):
        self.cleanups += 1

    def apply(self, _documents):
        pass

    def call(self, *_args):
        return "scaled"

    def get(self, kind):
        if self.interrupt and kind == "events":
            raise KeyboardInterrupt("during final evidence collection")
        if kind != "pods":
            return {"items": []}
        self.pod_reads += 1
        return {
            "items": [
                {
                    "metadata": {
                        "labels": {"app": role},
                        "name": role + "-abc-def",
                        "uid": role + ("-new" if self.replace_at and self.pod_reads >= self.replace_at else ""),
                    },
                    "status": {
                        "containerStatuses": [
                            {"imageID": "sha256:original", "restartCount": int(self.restart and self.log_reads > 0)}
                        ]
                    },
                }
                for role in ("ws-server", "vessel-1", "vessel-coordinator")
            ]
        }

    def logs(self, role, *, pod=None):
        self.log_reads += 1
        if pod is not None and pod != role + "-abc-def":
            raise AssertionError("Logs were not bound to the checked pod")
        text = self.content.get(role, "hub log")
        return text.replace('{"kind": "completed"}', "") if self.timeout and self.metric_reads == 0 else text

    def resources(self, **_kwargs):
        self.metric_reads += 1
        if self.timeout and self.metric_reads == 1:
            raise subprocess.TimeoutExpired("metrics", 20)
        return copy.deepcopy(self.samples[0])


class SafetyTests(unittest.TestCase):
    def test_invalid_completion_receipts_do_not_crash_resume_or_reference_external_artifacts(self):
        for marker in ("{", "[]", '{"signature":"same","cleanup_verified":true,"artifacts":[]}'):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "completion.json").write_text(marker)
                self.assertFalse(runner.verified(root, "same"))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            outside = root / "external.json"
            outside.write_text("evidence outside this attempt")
            attempt = root / "attempt-01"
            save(
                attempt / "completion.json",
                {
                    "signature": "same",
                    "cleanup_verified": True,
                    "artifacts": {"../external.json": hashlib.sha256(outside.read_bytes()).hexdigest()},
                },
            )
            self.assertFalse(runner.verified(attempt, "same"))

    def test_snapshot_retry_discards_abandoned_temporary_files_but_preserves_verified_assets(
        self,
    ):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, frozen = root / "live", root / "private"
            source.mkdir()
            frozen.mkdir()
            (source / "feature.bin").write_bytes(b"original")
            expected = {"feature.bin": hashlib.sha256(b"original").hexdigest()}
            staging = frozen / ".snapshot-staging"
            staging.mkdir()
            (staging / "interrupted-copy").write_bytes(b"partial" * 100)
            (frozen / "feature.bin").write_bytes(b"original")
            (source / "feature.bin").unlink()
            copy_verified(source, frozen, expected)
            self.assertEqual((frozen / "feature.bin").read_bytes(), b"original")
            self.assertFalse(staging.exists())

    def test_empty_metric_entry_does_not_hide_a_populated_entry_in_the_same_window(
        self,
    ):
        from measurements import resource_summary

        _settings, _partition, _logs, samples = result_inputs()
        filled = copy.deepcopy(samples[0]["items"][0])
        role = filled["metadata"]["name"].rsplit("-", 2)[0]
        empty = {**filled, "containers": []}
        with self.assertRaisesRegex(ValueError, "Missing resource samples"):
            resource_summary([{"items": [empty]}], [role])
        self.assertEqual(
            resource_summary([{"items": [empty, filled]}], [role])["roles"][role]["windows"],
            1,
        )

    def test_reasoned_authorization_denial_keeps_the_reason_and_resource(self):
        cluster = Mock()
        cluster.call.side_effect = CommandError(
            ["kubectl"],
            subprocess.CompletedProcess([], 1, "no - metrics access denied\n", ""),
        )
        with self.assertRaisesRegex(RuntimeError, "list pods.metrics.k8s.io: metrics access denied"):
            authorized(cluster, "list", "pods.metrics.k8s.io")

    def test_completion_clock_uses_event_time_or_terminal_observation(self):
        observed = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
        self.assertEqual(runner.completion_time('{"kind": "completed"}', observed), observed)
        content = 'event {"kind": "completed", "at": 1000}'
        self.assertEqual(
            runner.completion_time(content, observed),
            dt.datetime.fromtimestamp(1000, dt.UTC),
        )

    def test_missing_coordinator_metrics_require_windows_overlapping_training(self):
        for overlap in (True, False):
            with self.subTest(overlap=overlap), tempfile.TemporaryDirectory() as folder:
                cluster = FixtureCluster()
                original = cluster.resources

                def metrics(overlap=overlap, cluster=cluster, original=original, **kwargs):
                    sample = original(**kwargs)
                    if cluster.metric_reads == 1:
                        sample["items"] = [
                            p for p in sample["items"] if not p["metadata"]["name"].startswith("vessel-coordinator-")
                        ]
                    else:
                        self.assertGreater(kwargs["timeout"], 0)
                        for item in sample["items"]:
                            item["timestamp"] = (
                                dt.datetime.now(dt.UTC).isoformat() if overlap else "2000-01-01T00:00:00Z"
                            )
                    return sample

                cluster.resources = metrics
                output = Path(folder) / "attempt"
                if overlap:
                    self.assertEqual(self.execute(cluster, output)["status"], "passed")
                    self.assertEqual(cluster.metric_reads, 2)
                else:
                    with (
                        patch.object(runner.time, "monotonic", side_effect=itertools.count()),
                        self.assertRaisesRegex(ValueError, "Missing resource samples for: vessel-coordinator"),
                    ):
                        self.execute(cluster, output)
                    self.assertGreater(cluster.metric_reads, 1)
                    self.assertLess(cluster.metric_reads, 30)
                self.assertEqual(cluster.cleanups, 2)

    def test_private_feature_copy_survives_changes_to_live_preparation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, frozen = root / "live", root / "private"
            source.mkdir()
            (source / "feature.bin").write_bytes(b"original feature")
            expected = {"feature.bin": hashlib.sha256(b"original feature").hexdigest()}
            copy_verified(source, frozen, expected)
            (source / "feature.bin").write_bytes(b"reprepared feature")
            self.assertEqual((frozen / "feature.bin").read_bytes(), b"original feature")
            with self.assertRaisesRegex(ValueError, "changed before snapshot"):
                copy_verified(source, root / "rejected", expected)
            self.assertFalse((root / "rejected/feature.bin").exists())

    def test_source_identity_ignores_projected_tokens_but_tracks_effective_commands(self):
        settings, _, _, _ = result_inputs()
        value = snapshot()
        initial = source_signature(settings, value)
        spec = value["source_templates"]["vessel-1"]
        spec["volumes"] = [{"name": "kube-api-access-random", "projected": {}}]
        spec["containers"][0].setdefault("volumeMounts", []).append(
            {"name": "kube-api-access-random", "mountPath": "/var/run/secrets/kubernetes.io/serviceaccount"}
        )
        self.assertEqual(source_signature(settings, value), initial)
        spec["containers"][0]["command"] = ["changed-command"]
        self.assertNotEqual(source_signature(settings, value), initial)

    def test_workloads_read_the_verified_copy_instead_of_the_live_host_directory(self):
        settings, _, _, _ = result_inputs()
        value = snapshot()
        data, _ = runner.prepare(value, 1, 17)
        docs = documents(settings, value, data, 1, "snapshot-test")
        client = next(doc for doc in docs if doc["kind"] == "Deployment" and doc["metadata"]["name"] == "vessel-1")
        spec = client["spec"]["template"]["spec"]
        self.assertIn({"name": "source-data", "emptyDir": {}}, spec["volumes"])
        self.assertNotIn("live-source-data", {m["name"] for m in spec["containers"][0]["volumeMounts"]})
        self.assertIn(
            {"name": "source-data", "mountPath": "/snapshot-output"}, spec["initContainers"][0]["volumeMounts"]
        )

    def test_noninteger_client_counts_are_rejected(self):
        settings, _, _, _ = result_inputs()
        settings["clients"] = [1.0, 3, 6, 12]
        with self.assertRaisesRegex(ValueError, "participant counts"):
            runner.validate(settings)

    def test_local_manifest_write_failure_never_publishes_an_attempt(self):
        cluster = FixtureCluster()
        real_save = runner.save

        def save_inputs(path, value):
            if path.name == "manifests.json":
                raise OSError("disk full")
            return real_save(path, value)

        with tempfile.TemporaryDirectory() as folder, patch.object(runner, "save", side_effect=save_inputs):
            output = Path(folder) / "attempt-01"
            with self.assertRaisesRegex(OSError, "disk full"):
                self.execute(cluster, output)
            self.assertFalse(output.exists())
            self.assertEqual(cluster.cleanups, 0)

    def test_metrics_with_distinct_windows_are_not_counted_as_synchronized(self):
        from measurements import resource_summary

        _, _, _, samples = result_inputs()
        samples[0]["items"][0]["window"] = "30s"
        summary = resource_summary(samples, ["ws-server", "vessel-coordinator", "vessel-1"])
        self.assertEqual(summary["synchronised_complete_windows"], 0)
        self.assertEqual(summary["aggregate_mean_cpu_cores"], 1.5)
        for item in samples[0]["items"]:
            item["window"] = "0.5m" if item is samples[0]["items"][0] else "30s"
        self.assertEqual(
            resource_summary(samples, ["ws-server", "vessel-coordinator", "vessel-1"])["synchronised_complete_windows"],
            1,
        )

    def test_dataset_probe_is_bound_to_a_stable_pod_and_its_mount(self):
        for mode in ("stable", "rollout", "replaced", "hub-missing", "hub-cached", "study-metrics", "node-metrics"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                cluster = Mock()
                cluster.app = Path(folder)
                (cluster.app / "src/harbour_fl").mkdir(parents=True)
                save(cluster.app / "recovery.json", {})
                cluster.settings = {**runner.load(runner.ROOT / "settings.json"), "source_namespace": "source"}
                cluster.owned_namespace.return_value = None
                reads, probes = {}, []

                def get(kind, *args, mode=mode, reads=reads, **_kwargs):
                    if kind.startswith("deployment/"):
                        return {
                            "metadata": {"generation": 1},
                            "spec": {
                                "replicas": 1,
                                "template": {"spec": {"volumes": [{"hostPath": {"path": "/new"}}]}},
                            },
                            "status": {
                                "observedGeneration": 1,
                                "replicas": 1,
                                "readyReplicas": 1,
                                "updatedReplicas": 0 if mode == "rollout" else 1,
                                "availableReplicas": 1,
                            },
                        }
                    if kind == "pods":
                        role = args[1].removeprefix("app=")
                        reads[role] = reads.get(role, 0) + 1
                        return {
                            "items": [
                                {
                                    "metadata": {
                                        "name": role + "-fixed",
                                        "uid": role
                                        + ("-new" if mode == "replaced" and reads[role] > 1 else "-original"),
                                    },
                                    "spec": {
                                        "nodeName": "other"
                                        if role == "ws-server" and mode.startswith("hub-")
                                        else "node",
                                        "containers": [
                                            {
                                                "image": "app" if role != "ws-server" else "hub",
                                                "env": [
                                                    {
                                                        "name": "RUNNER_MODULE",
                                                        "value": "module",
                                                    },
                                                    {
                                                        "name": "MODULES_PATHS",
                                                        "value": "/modules",
                                                    },
                                                ],
                                                "volumeMounts": [
                                                    {
                                                        "name": "data",
                                                        "mountPath": "/data",
                                                    }
                                                ],
                                            }
                                        ],
                                        "volumes": [
                                            {
                                                "name": "data",
                                                "hostPath": {"path": "/old"},
                                            }
                                        ],
                                    },
                                    "status": {
                                        "phase": "Running",
                                        "containerStatuses": [
                                            {
                                                "ready": True,
                                                "imageID": "app" if role != "ws-server" else "hub",
                                            }
                                        ],
                                    },
                                }
                            ]
                        }
                    if kind.startswith("node/"):
                        return {
                            "metadata": {"labels": {"kubernetes.io/hostname": "node"}},
                            "status": {
                                "nodeInfo": {},
                                "allocatable": {"cpu": "20", "memory": "128Gi"},
                                "images": [{"names": ["hub"]}] if mode == "hub-cached" else [],
                            },
                        }
                    if kind == "storageclasses":
                        return {
                            "items": [
                                {
                                    "metadata": {
                                        "name": "storage",
                                        "annotations": {"storageclass.kubernetes.io/is-default-class": "true"},
                                    },
                                    "reclaimPolicy": "Delete",
                                }
                            ]
                        }
                    raise AssertionError(kind)

                def call(*args, probes=probes, mode=mode, **_kwargs):
                    if args[0] == "auth":
                        if (mode == "study-metrics" and "pods.metrics.k8s.io" in args) or (
                            mode == "node-metrics" and "nodes.metrics.k8s.io" in args
                        ):
                            return "no - metrics access denied"
                        return "yes"
                    if args[0] == "exec":
                        probes.append(args)
                        return json.dumps(
                            {
                                "root": "/data/dataset",
                                "code_hashes": {},
                                "model_signature": "model",
                                "application_root": "/opt/application",
                                "preparation": {},
                            }
                        )
                    return json.dumps({"items": [{}]})

                cluster.get.side_effect = get
                cluster.call.side_effect = call
                with patch.dict(
                    sys.modules,
                    {"harbour_fl.workload": SimpleNamespace(signature=lambda: "model")},
                ):
                    if mode in ("stable", "hub-cached"):
                        result = preflight(cluster)
                        self.assertEqual(result["dataset_host_path"], "/old/dataset")
                        self.assertEqual(probes[0][1], "pod/vessel-1-fixed")
                    elif mode in ("rollout", "replaced"):
                        with self.assertRaisesRegex(RuntimeError, "rollout is incomplete|Source pod changed"):
                            preflight(cluster)
                        self.assertEqual(len(probes), int(mode == "replaced"))
                    elif mode == "hub-missing":
                        with self.assertRaisesRegex(RuntimeError, "Image for ws-server is not verified on node"):
                            preflight(cluster)
                        self.assertEqual(probes, [])
                    else:
                        with self.assertRaisesRegex(RuntimeError, "metrics.k8s.io: metrics access denied"):
                            preflight(cluster)

    def test_namespace_creation_cannot_adopt_an_existing_namespace(self):
        with tempfile.TemporaryDirectory() as folder:
            cluster = Cluster.__new__(Cluster)
            cluster.namespace = "et-benchmark-scaling-test"
            cluster.namespace_journal = Path(folder) / "namespace-owner.json"
            cluster.call = Mock(side_effect=RuntimeError("AlreadyExists"))
            document = {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": cluster.namespace}}
            with self.assertRaisesRegex(RuntimeError, "AlreadyExists"):
                cluster.apply([document])
            self.assertEqual(cluster.call.call_args.args[:2], ("create", "-f"))
            self.assertIsNone(json.loads(cluster.namespace_journal.read_text())["uid"])
            cluster.owned_namespace = Mock(
                return_value={"uid": "unrelated", "labels": {"edge-toolkit.local/attempt": "someone-else"}}
            )
            with self.assertRaisesRegex(RuntimeError, "does not belong"):
                cluster.cleanup()
            self.assertEqual(cluster.call.call_count, 1)

    def test_failed_workload_apply_retains_the_created_namespace_uid_for_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            cluster = Cluster.__new__(Cluster)
            cluster.namespace = "et-benchmark-scaling-test"
            cluster.namespace_journal = Path(folder) / "namespace-owner.json"
            cluster.call = Mock(
                side_effect=[json.dumps({"metadata": {"uid": "created"}}), RuntimeError("apply failed"), "{}"]
            )
            doc = {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": cluster.namespace}}
            with self.assertRaisesRegex(RuntimeError, "apply failed"):
                cluster.apply([doc])
            owner = json.loads(cluster.namespace_journal.read_text())
            self.assertEqual(owner["uid"], "created")
            cluster.owned_namespace = Mock(
                side_effect=[
                    {
                        "uid": "created",
                        "resourceVersion": "1",
                        "labels": {"edge-toolkit.local/attempt": owner["token"]},
                    },
                    None,
                ]
            )
            cluster.cleanup()
            self.assertEqual(json.loads(cluster.call.call_args.kwargs["stdin"])["preconditions"]["uid"], "created")

    def test_cluster_wide_capacity_permission_is_checked_separately(self):
        cluster = Mock()
        cluster.call.side_effect = lambda *args: "no" if "--all-namespaces" in args else "yes"
        with self.assertRaisesRegex(RuntimeError, "list pods --all-namespaces"):
            preflight(cluster)
        self.assertIn(unittest.mock.call("auth", "can-i", "list", "pods"), cluster.call.call_args_list)
        cluster.get.assert_not_called()

    def test_any_unverified_attempt_blocks_a_different_case(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            save(output / "c06-r01/attempt-01/report.json", {"cleanup_verified": False})
            with (
                patch.object(sys, "argv", ["runner", "smoke", "--case", "1", "--output", folder]),
                patch.object(runner, "Cluster", return_value=Mock()),
                patch.object(runner, "locks", return_value=contextlib.nullcontext()),
                patch.object(runner, "preflight") as preflight,
                patch.object(runner, "execute") as execute,
            ):
                with self.assertRaisesRegex(RuntimeError, "Unverified cleanup"):
                    runner.main()
                preflight.assert_not_called()
                execute.assert_not_called()

    def test_reuse_signature_is_bound_to_the_resolved_cluster(self):
        cluster = Mock()
        cluster.target.return_value = {"server": "server", "cluster_uid": "first", "namespace": "study"}
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(sys, "argv", ["runner", "check", "--output", folder]),
            patch.object(runner, "Cluster", return_value=cluster),
            patch.object(runner, "locks", return_value=contextlib.nullcontext()),
            patch.object(runner, "preflight", return_value={"identity": "source"}),
            patch.object(runner, "prepare", return_value=({}, {})),
            patch.object(runner, "documents", return_value=[]),
            patch.object(runner, "tool_identity", return_value="tool"),
        ):
            runner.main()
            original = (Path(folder) / "inputs.json").read_bytes()
            self.assertEqual(cluster.target.call_count, 1)
            runner.main()
            cluster.target.return_value = {**cluster.target.return_value, "cluster_uid": "second"}
            with self.assertRaisesRegex(RuntimeError, "Inputs changed"):
                runner.main()
            self.assertEqual((Path(folder) / "inputs.json").read_bytes(), original)

    def test_readiness_observed_after_deadline_is_rejected(self):
        for kind in ("client", "rollout"):
            with self.subTest(kind=kind):
                clock = [0.0]
                cluster = Mock()

                def response(*_args, kind=kind, clock=clock, **kwargs):
                    self.assertEqual(kwargs["timeout"], 1)
                    clock[0] = 1
                    return (
                        "validation pairs"
                        if kind == "client"
                        else {
                            "metadata": {"generation": 1},
                            "status": {"observedGeneration": 1, "availableReplicas": 1},
                        }
                    )

                cluster.logs.side_effect = response
                cluster.get.side_effect = response
                with (
                    patch.object(runner.time, "monotonic", side_effect=lambda clock=clock: clock[0]),
                    self.assertRaises(TimeoutError),
                ):
                    (runner.wait_clients if kind == "client" else runner.wait_rollouts)(
                        cluster, ["vessel-1"], timeout=1
                    )

    def test_empty_verified_results_remove_stale_curves(self):
        from measurements import curves

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            for name in ("scaling.csv", "scaling.svg"):
                (output / name).write_text("stale result")
            curves(output, {})
            self.assertFalse((output / "scaling.csv").exists())
            self.assertFalse((output / "scaling.svg").exists())

    def execute(self, cluster, folder):
        with (
            patch.object(runner, "wait_rollouts"),
            patch.object(runner, "wait_clients"),
            patch.object(runner.time, "sleep"),
        ):
            return runner.execute(cluster, cluster.settings, snapshot(), folder, "point", 1, 1)

    def test_restart_on_completion_and_replacement_during_final_collection_are_rejected(self):
        for options in ({"restart": True}, {"replace_at": 3}, {"replace_at": 5}, {"replace_at": 8}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as folder:
                cluster = FixtureCluster(**options)
                path = Path(folder) / "attempt"
                with self.assertRaisesRegex(RuntimeError, "restarted|replaced"):
                    self.execute(cluster, path)
                self.assertEqual(cluster.cleanups, 2)
                self.assertEqual(json.loads((path / "report.json").read_text())["status"], "failed")

    def test_interruption_in_final_log_collection_still_cleans_the_namespace(self):
        with tempfile.TemporaryDirectory() as folder:
            cluster = FixtureCluster(interrupt=True)
            with self.assertRaises(KeyboardInterrupt):
                self.execute(cluster, Path(folder) / "attempt")
            self.assertEqual(cluster.cleanups, 2)

    def test_metrics_timeout_is_recorded_and_other_samples_can_qualify(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "attempt"
            report = self.execute(FixtureCluster(timeout=True), path)
            self.assertEqual(report["status"], "passed")
            self.assertIn("timed out", json.loads((path / "resource-samples.json").read_text())["errors"][0])

    def test_restartable_init_requests_include_overlap_and_order(self):
        for initial, expected in (
            ([{"cpu": 3, "sidecar": True}], 7),
            ([{"cpu": 10}, {"cpu": 3, "sidecar": True}, {"cpu": 8}], 11),
        ):
            pod = {
                "metadata": {"namespace": "operational"},
                "spec": {
                    "nodeName": "node",
                    "containers": [{"resources": {"requests": {"cpu": "4"}}}],
                    "initContainers": [
                        {
                            "resources": {"requests": {"cpu": str(value["cpu"])}},
                            **({"restartPolicy": "Always"} if value.get("sidecar") else {}),
                        }
                        for value in initial
                    ],
                },
            }
            self.assertEqual(request_totals([pod], "node", "study")["cpu"], expected)

    def test_pod_level_requests_override_only_the_specified_resource(self):
        pod = {
            "metadata": {"namespace": "other"},
            "spec": {
                "nodeName": "node",
                "resources": {"requests": {"cpu": "10"}},
                "overhead": {"cpu": "1"},
                "containers": [{"resources": {"requests": {"cpu": "2", "memory": "1Gi"}}}],
            },
        }
        self.assertEqual(request_totals([pod], "node", "study"), {"cpu": 11, "memory": 1024})

    def test_foreign_pod_metrics_cannot_qualify_the_bound_role(self):
        cluster = FixtureCluster()
        cluster.samples[0]["items"][0]["metadata"]["name"] = "ws-server-debug"
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(runner.time, "monotonic", side_effect=itertools.count()),
            self.assertRaisesRegex(ValueError, "Missing resource"),
        ):
            self.execute(cluster, Path(folder) / "attempt")
        self.assertEqual(cluster.cleanups, 2)

    def test_qualified_logs_are_not_replaced_by_a_later_read(self):
        cluster = FixtureCluster()
        original = cluster.logs
        counts = {}

        def logs(role, **kwargs):
            counts[role] = counts.get(role, 0) + 1
            return original(role, **kwargs) if counts[role] <= (2 if role == "vessel-coordinator" else 1) else "rotated"

        cluster.logs = logs
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "attempt"
            report = self.execute(cluster, path)
            self.assertEqual(report["status"], "passed")
            self.assertIn("completed round", (path / "vessel-1.log").read_text())
            self.assertEqual(counts["vessel-1"], 1)

    def test_target_failure_does_not_publish_an_attempt(self):
        cluster = FixtureCluster()
        cluster.target = Mock(side_effect=TimeoutError("target lookup"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "attempt"
            with self.assertRaisesRegex(TimeoutError, "target lookup"):
                self.execute(cluster, path)
            self.assertFalse(path.exists())
            self.assertEqual(cluster.cleanups, 0)

    def test_namespace_deletion_uses_both_server_preconditions(self):
        cluster = Cluster.__new__(Cluster)
        cluster.namespace = "et-benchmark-scaling-test"
        identity = {"uid": "owned-uid", "resourceVersion": "42", "labels": {"edge-toolkit.local/attempt": "token"}}
        cluster.namespace_owner = {"uid": "owned-uid", "token": "token"}
        cluster.owned_namespace = Mock(side_effect=[identity, None])
        cluster.call = Mock()
        cluster.cleanup()
        self.assertEqual(
            json.loads(cluster.call.call_args.kwargs["stdin"])["preconditions"],
            {"uid": "owned-uid", "resourceVersion": "42"},
        )
        cluster.owned_namespace = Mock(return_value=identity)
        cluster.call = Mock(side_effect=RuntimeError("409 precondition failed"))
        with self.assertRaisesRegex(RuntimeError, "precondition"):
            cluster.cleanup()

    def test_nested_dataset_mount_is_rejected(self):
        spec = {
            "containers": [{"volumeMounts": [{"name": "data", "mountPath": "/data"}]}],
            "volumes": [{"name": "data", "hostPath": {"path": "/mnt/data"}}],
        }
        self.assertEqual(dataset_host_path("/data/croatia", spec), "/mnt/data/croatia")
        spec["containers"][0]["volumeMounts"].append({"name": "features", "mountPath": "/data/croatia/features"})
        with self.assertRaisesRegex(RuntimeError, "Nested dataset"):
            dataset_host_path("/data/croatia", spec)

    def test_completion_after_deadline_is_rejected_and_cleaned(self):
        cluster = FixtureCluster()
        cluster.settings["job_timeout_seconds"] = 1
        clock = [0]
        resources = cluster.resources

        def metrics():
            clock[0] = 2
            return resources()

        cluster.resources = metrics
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(runner.time, "monotonic", side_effect=lambda: clock[0]),
            self.assertRaisesRegex(TimeoutError, "overall timeout"),
        ):
            self.execute(cluster, Path(folder) / "attempt")
        self.assertEqual(cluster.cleanups, 2)

    def test_generated_custom_output_is_not_a_source_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "runner.py").write_text("source")
            save(root / "scratch-run/inputs.json", {"credential": "private"})
            self.assertEqual(sources(root), [root / "runner.py"])

    def test_custom_evidence_does_not_change_identity_but_source_changes_do(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "runner.py").write_text("source")
            (root / "study_support.py").write_text("support")
            source = root / "controller"
            source.mkdir()
            (source / "implementation.py").write_text("controller")
            with patch("common.controller_source", return_value=source):
                before = tool_identity(root)
                save(root / "scratch-run/inputs.json", {"credential": "private"})
                self.assertEqual(tool_identity(root), before)
                (root / "runner.py").write_text("changed source")
                self.assertNotEqual(tool_identity(root), before)

    def test_abbreviated_settings_option_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.parser().parse_args(["check", "--sett", "local.json"])

    def test_subresource_authorization_uses_explicit_kubectl_option(self):
        for denied in ("scale", "log"):
            cluster = Mock()
            cluster.call.side_effect = lambda *args, denied=denied: "no" if "--subresource=" + denied in args else "yes"
            with self.assertRaisesRegex(RuntimeError, "permission missing"):
                preflight(cluster)
            self.assertIn(
                unittest.mock.call(
                    "auth",
                    "can-i",
                    "patch" if denied == "scale" else "get",
                    "deployments" if denied == "scale" else "pods",
                    "--subresource=" + denied,
                ),
                cluster.call.call_args_list,
            )

    def test_nonzero_authorization_denial_names_the_subresource(self):
        cluster = Mock()

        def call(*args):
            if "--subresource=scale" in args:
                raise CommandError(["kubectl"], subprocess.CompletedProcess([], 1, "no\n", ""))
            return "yes"

        cluster.call.side_effect = call
        with self.assertRaisesRegex(RuntimeError, "Scaling permission missing: patch deployments/scale"):
            preflight(cluster)

    def test_cleanup_cannot_attest_another_cluster_or_unknown_target(self):
        for identity in (None, {"cluster_uid": "different"}, {"cluster_uid": "same"}):
            with self.subTest(identity=identity), tempfile.TemporaryDirectory() as folder:
                output = Path(folder)
                attempt = output / "point/attempt-01"
                save(attempt / "report.json", {"status": "failed", "cleanup_verified": False})
                if identity is not None:
                    save(attempt / "run-target.json", identity)
                cluster = Mock()
                cluster.target.return_value = {"cluster_uid": "same"}
                with (
                    patch.object(sys, "argv", ["runner", "cleanup", "--output", str(output)]),
                    patch.object(runner, "Cluster", return_value=cluster),
                    patch.object(runner, "locks", return_value=contextlib.nullcontext()),
                ):
                    if identity == cluster.target.return_value:
                        runner.main()
                        self.assertTrue(json.loads((attempt / "report.json").read_text())["cleanup_verified"])
                    else:
                        with self.assertRaisesRegex(RuntimeError, "target differs or is unknown"):
                            runner.main()
                        cluster.cleanup.assert_not_called()

    def test_help_and_empty_smoke_selection_do_not_access_application_or_cluster(self):
        for values in (["--help"], ["check", "--help"]):
            with (
                patch.object(sys, "argv", ["cli", *values]),
                patch.object(cli, "load", side_effect=AssertionError("application prerequisite accessed")),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                cli.main()
                self.assertIn("--case", output.getvalue())
        with patch.object(sys, "argv", ["runner", "smoke", "--case", "3"]), patch.object(runner, "Cluster") as cluster:
            with self.assertRaisesRegex(ValueError, "Smoke supports"):
                runner.main()
            cluster.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            settings = copy.deepcopy(FixtureCluster().settings)
            settings["clients"] = [1, 3, 6, 12]
            path = Path(folder) / "settings.json"
            save(path, settings)
            with (
                patch.object(sys, "argv", ["runner", "run", "--case", "24", "--settings", str(path)]),
                patch.object(runner, "Cluster") as cluster,
            ):
                with self.assertRaisesRegex(ValueError, "matches no jobs"):
                    runner.main()
                cluster.assert_not_called()

    def test_results_archive_excludes_env_bearing_inputs_and_manifests(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("inputs.json", "manifests.json", "pods-final.json", "report.json"):
                save(
                    root / "results/run" / name,
                    {"env": "credential-should-not-export"} if name != "report.json" else {"status": "passed"},
                )
            for name in ("README.md", "settings.json", "settings-per-client.json"):
                (root / name).write_text("{}")
            with (
                patch.object(package, "ROOT", root),
                patch.object(package, "tool_identity", return_value="identity"),
                patch.object(sys, "argv", ["package", "--results"]),
            ):
                package.main()
            with tarfile.open(next((root / "packages").glob("*.tar.gz"))) as archive:
                self.assertTrue(any(Path(name).name == "report.json" for name in archive.getnames()))
                for member in archive.getmembers():
                    self.assertNotIn(b"credential-should-not-export", archive.extractfile(member).read())

    def test_source_archive_excludes_evidence_in_a_custom_output_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("README.md", "runner.py", "settings.json", "settings-per-client.json", "study_support.py"):
                (root / name).write_text("source")
            save(root / "scratch-run/inputs.json", {"env": "private-env-value"})
            controller = root / "support"
            controller.mkdir()
            (controller / "engine.py").write_text("controller source")
            with (
                patch.object(package, "ROOT", root),
                patch.object(package, "tool_identity", return_value="identity"),
                patch.object(package, "sources", side_effect=lambda: sources(root)),
                patch.object(package, "support_source", return_value=root / "study_support.py"),
                patch.object(package, "controller_source", return_value=controller),
                patch.object(sys, "argv", ["package"]),
            ):
                package.main()
            with tarfile.open(next((root / "packages").glob("*.tar.gz"))) as archive:
                names = [name.split("/", 1)[1] for name in archive.getnames()]
                self.assertIn("settings.json", names)
                self.assertIn("settings-per-client.json", names)
                self.assertIn("study_support.py", names)
                self.assertIn("controller/src/engine.py", names)
                self.assertFalse(any("scratch-run" in name for name in names))
