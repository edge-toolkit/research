"""Create an isolated single-hub namespace from the already deployed host images."""

import copy
import hashlib
import json
import time
import uuid
from pathlib import Path

from common import ROOT, CommandError, command, digest, save
from study_support import adapter, application, kubectl_prefix, source_identity

OWNER = "harbour-fl-scaling"
LABEL = "edge-toolkit.local/study"
ATTEMPT_LABEL = "edge-toolkit.local/attempt"


def locate(settings):
    """Resolve the maintained Harbour-FL application and its shared study lock location."""
    app = application(settings, ROOT)
    return app


class Cluster:
    """Keep every Kubernetes request on an explicit context and namespace."""

    def __init__(self, settings):
        self.settings = settings
        self.app = locate(settings)
        self.namespace = settings["namespace"]
        if self.namespace == settings["source_namespace"] or not self.namespace.startswith("et-benchmark-scaling"):
            raise ValueError("Scaling requires its own et-benchmark-scaling namespace")
        self.prefix = kubectl_prefix(settings)
        self.namespace_owner = None
        self.namespace_journal = None
        self.integration, self.adapter = adapter(
            self.app, ROOT, {**settings, "namespace": self.namespace}, workers=("vessel-1",)
        )

    def call(self, *args, source=False, timeout=60, stdin=None):
        """Bound commands and never change the user's global kubectl context."""
        ns = self.settings["source_namespace"] if source else self.namespace
        return command([*self.prefix, "-n", ns, *args], timeout=timeout, stdin=stdin)

    def get(self, *args, **kwargs):
        """Decode one Kubernetes JSON response."""
        return json.loads(self.call("get", *args, "-o", "json", **kwargs))

    def apply(self, documents):
        """Apply only generated objects carrying the dedicated namespace."""
        for doc in documents:
            if doc["kind"] == "Namespace" and doc["metadata"]["name"] != self.namespace:
                raise ValueError("Refusing a different namespace")
            if doc["kind"] != "Namespace" and doc["metadata"].get("namespace") != self.namespace:
                raise ValueError("Refusing an object outside the scaling namespace")
        namespaces = [doc for doc in documents if doc["kind"] == "Namespace"]
        if len(namespaces) != 1 or self.namespace_journal is None:
            raise RuntimeError("Namespace creation requires an attempt ownership journal")
        namespace = copy.deepcopy(namespaces[0])
        self.namespace_owner = {"token": uuid.uuid4().hex, "uid": None}
        namespace["metadata"].setdefault("labels", {}).update(
            {LABEL: OWNER, ATTEMPT_LABEL: self.namespace_owner["token"]}
        )
        save(self.namespace_journal, self.namespace_owner)
        created = json.loads(self.call("create", "-f", "-", "-o", "json", stdin=json.dumps(namespace)))
        self.namespace_owner["uid"] = created["metadata"]["uid"]
        save(self.namespace_journal, self.namespace_owner)
        # Client-side apply fails with "metadata.annotations: Too long: may not be more than 262144 bytes"
        # when it duplicates a large dataset ConfigMap in the last-applied annotation.
        return self.call(
            "apply",
            "--server-side",
            "--field-manager=harbour-fl-scaling",
            "-f",
            "-",
            stdin=json.dumps(
                {"apiVersion": "v1", "kind": "List", "items": [doc for doc in documents if doc["kind"] != "Namespace"]}
            ),
        )

    def owned_namespace(self):
        """Refuse to adopt or delete an unrelated namespace."""
        raw = self.call("get", "namespace/" + self.namespace, "--ignore-not-found", "-o", "json")
        metadata = json.loads(raw)["metadata"] if raw.strip() else None
        if metadata and metadata.get("labels", {}).get(LABEL) != OWNER:
            raise RuntimeError("Namespace already exists without the scaling ownership label")
        return metadata

    def cleanup(self):
        """Remove only the owned experiment namespace, including its independent storage claim."""
        identity = self.owned_namespace()
        if identity:
            owner = self.namespace_owner
            if (
                not owner
                or identity.get("labels", {}).get(ATTEMPT_LABEL) != owner["token"]
                or (owner.get("uid") is not None and identity["uid"] != owner["uid"])
            ):
                raise RuntimeError("Namespace does not belong to this attempt; cleanup remains unverified")
            self.call(
                "delete",
                "--raw",
                "/api/v1/namespaces/" + self.namespace,
                "-f",
                "-",
                stdin=json.dumps(
                    {
                        "apiVersion": "v1",
                        "kind": "DeleteOptions",
                        "preconditions": {"uid": identity["uid"], "resourceVersion": identity["resourceVersion"]},
                    }
                ),
            )
            deadline = time.monotonic() + 180
            while current := self.owned_namespace():
                if current["uid"] != identity["uid"]:
                    raise RuntimeError("Scaling namespace was replaced during cleanup")
                if time.monotonic() >= deadline:
                    raise RuntimeError("Scaling namespace cleanup has not completed")
                time.sleep(1)

    def logs(self, name, *, pod=None, timeout=30):
        """Collect the current role's complete log with Kubernetes timestamps."""
        return self.call("logs", "pod/" + pod if pod else "deployment/" + name, "--timestamps=true", timeout=timeout)

    def target(self):
        """Resolve the configured API server and cluster UID before attesting cleanup."""
        configuration = json.loads(command([*self.prefix, "config", "view", "-o", "json"]))
        context = next(row["context"] for row in configuration["contexts"] if row["name"] == self.settings["context"])
        server = next(
            row["cluster"]["server"] for row in configuration["clusters"] if row["name"] == context["cluster"]
        )
        return {
            "server": server,
            "cluster_uid": self.get("namespace/kube-system")["metadata"]["uid"],
            "namespace": self.namespace,
        }

    def resources(self, *, timeout=20):
        """Read CPU cores and working-set memory through the existing Metrics API."""
        return json.loads(
            self.call(
                "get", "--raw", "/apis/metrics.k8s.io/v1beta1/namespaces/" + self.namespace + "/pods", timeout=timeout
            )
        )


def verify_application(snapshot, templates):
    """Require the deployed model and application source to match the current Harbour-FL checkout."""
    expected = snapshot["expected_code_hashes"]
    actual = snapshot["code_hashes"]
    changed = sorted(name for name in expected.keys() | actual.keys() if actual.get(name) != expected.get(name))
    if changed:
        raise RuntimeError("Deployed workload differs from current Harbour-FL source: " + ", ".join(changed))
    if snapshot["model_signature"] != snapshot["expected_model_signature"]:
        raise RuntimeError("Deployed pretrained model identity differs from the current application")
    root = Path(snapshot["application_root"])
    for role in ("vessel-1", "vessel-coordinator"):
        for mount in templates[role]["containers"][0].get("volumeMounts", []):
            target = Path(mount["mountPath"])
            if target == root or target.is_relative_to(root) or root.is_relative_to(target):
                raise RuntimeError("Application must be bundled in the image, not volume-mounted: " + role)


def dataset_host_path(root, spec):
    """Map one plain dataset binding; nested mounts need a separately reproducible topology."""
    root = Path(root)
    mounts = spec["containers"][0].get("volumeMounts", [])
    if any(Path(m["mountPath"]) != root and Path(m["mountPath"]).is_relative_to(root) for m in mounts):
        raise RuntimeError("Nested dataset mounts cannot be reproduced by the scaling study")
    enclosing = [m for m in mounts if root.is_relative_to(m["mountPath"])]
    if not enclosing:
        raise RuntimeError("Prepared data must have a dedicated hostPath mount")
    mount = max(enclosing, key=lambda m: len(m["mountPath"]))
    volume = next(v for v in spec["volumes"] if v["name"] == mount["name"])
    if "hostPath" not in volume or mount.get("subPath") or mount.get("subPathExpr"):
        raise RuntimeError("Dataset mount is not a supported plain hostPath; review an explicit read-only binding")
    return str(Path(volume["hostPath"]["path"]) / root.relative_to(mount["mountPath"]))


def authorized(cluster, verb, resource, *, all_namespaces=False):
    """Distinguish a denied authorization probe from a failed API request."""
    kind, separator, subresource = resource.partition("/")
    options = ["--subresource=" + subresource] if separator else []
    if all_namespaces:
        options.append("--all-namespaces")
    try:
        allowed = cluster.call("auth", "can-i", verb, kind, *options).strip()
    except CommandError as exc:
        answer = exc.stdout.strip()
        if exc.returncode != 1 or not (answer == "no" or answer.startswith("no - ")):
            raise
        allowed = answer
    if allowed != "yes":
        raise RuntimeError(
            f"Scaling permission missing: {verb} {resource}"
            + (" --all-namespaces" if all_namespaces else "")
            + (": " + allowed[5:] if allowed.startswith("no - ") else "")
        )


def source_pod(cluster, name):
    """Require a completed single-replica rollout and select its only running, ready pod."""
    deployment = cluster.get("deployment/" + name, source=True)
    status = deployment.get("status", {})
    if (
        deployment["spec"].get("replicas", 1) != 1
        or status.get("observedGeneration", 0) < deployment["metadata"]["generation"]
        or any(status.get(key, 0) != 1 for key in ("replicas", "updatedReplicas", "readyReplicas", "availableReplicas"))
        or status.get("unavailableReplicas", 0)
    ):
        raise RuntimeError("Source deployment rollout is incomplete: " + name)
    pods = cluster.get("pods", "-l", "app=" + name, source=True)["items"]
    if (
        len(pods) != 1
        or pods[0]["metadata"].get("deletionTimestamp")
        or pods[0].get("status", {}).get("phase") != "Running"
    ):
        raise RuntimeError("Source role is replacing or duplicated: " + name)
    statuses = pods[0]["status"].get("containerStatuses", [])
    if len(statuses) != 1 or not statuses[0].get("ready"):
        raise RuntimeError("Expected one ready source container: " + name)
    return deployment, pods[0]


def preflight(cluster):
    """Read source manifests and small dataset metadata without changing the original workloads."""
    for verb, resource in (
        ("get", "namespaces"),
        ("create", "namespaces"),
        ("delete", "namespaces"),
        ("create", "deployments"),
        ("get", "deployments"),
        ("patch", "deployments"),
        ("patch", "deployments/scale"),
        ("create", "services"),
        ("patch", "services"),
        ("create", "configmaps"),
        ("patch", "configmaps"),
        ("create", "persistentvolumeclaims"),
        ("patch", "persistentvolumeclaims"),
        ("get", "pods/log"),
        ("list", "pods"),
        ("list", "events"),
    ):
        authorized(cluster, verb, resource)
    authorized(cluster, "list", "pods", all_namespaces=True)
    templates = {}
    images = {}
    node = None
    selected = {}
    for name in ("ws-server", "vessel-1", "vessel-coordinator"):
        deployment, pod = source_pod(cluster, name)
        selected[name] = (deployment["metadata"]["generation"], pod)
        status = pod["status"]["containerStatuses"]
        if name == "vessel-1":
            node = pod["spec"]["nodeName"]
        templates[name] = pod["spec"]
        images[name] = {
            "image": pod["spec"]["containers"][0]["image"],
            "image_id": status[0]["imageID"],
        }
    if images["vessel-1"] != images["vessel-coordinator"]:
        raise RuntimeError("Coordinator and client must use the same application image")
    node_info = cluster.get("node/" + node)
    cached_images = {
        name.removeprefix("docker-pullable://").removeprefix("docker://").removeprefix("containerd://")
        for image in node_info["status"].get("images", [])
        for name in image.get("names", [])
    }
    for role, (_, pod) in selected.items():
        identity = images[role]["image_id"]
        normal = identity.removeprefix("docker-pullable://").removeprefix("docker://").removeprefix("containerd://")
        if pod["spec"]["nodeName"] != node and normal not in cached_images:
            raise RuntimeError(
                f"Image for {role} is not verified on node {node}; import that image onto the study node"
            )
    # A read-only query of the deployed loader catches unusable samples before a long training run.
    code = """import json,os,pathlib,csv,io,hashlib,harbour_fl
os.environ['HARBOUR_FL_MODEL_PROFILE']='pretrained'
from harbour_fl.data import _data_root,_load_rows
from harbour_fl.workload import signature
r=_data_root()
names=['class_summary.csv','feature_manifest.csv','test_manifest.csv']
names += [f'clients/vessel_{i}/{split}_manifest.csv' for i in (1,2,3) for split in ('train','val')]
m={name:(r/name).read_text() for name in names}
usable_ids={}
manifest_counts={}
for name in names:
    if name.endswith('_manifest.csv') and name!='feature_manifest.csv':
        records=list(csv.DictReader(io.StringIO(m[name])))
        usable=_load_rows(r,r/name)
        usable_ids[name]=[row['segment_id'] for row in usable]
        manifest_counts[name]={'rows':len(records),'usable_pairs':len(usable),'excluded':len(records)-len(usable)}
usable_sets={name:set(values) for name,values in usable_ids.items()}
assets={}
for name in names:
    if name.endswith('_manifest.csv') and name!='feature_manifest.csv':
        for row in csv.DictReader(io.StringIO(m[name])):
            if row['segment_id'] not in usable_sets[name]:
                continue
            for asset in (row['feature_path'],):
                if asset not in assets:
                    assets[asset]=hashlib.sha256((r/asset).read_bytes()).hexdigest()
source=pathlib.Path(harbour_fl.__file__).parent
code_hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*.py')}
print(json.dumps({'root':str(r),'manifests':m,'usable_ids':usable_ids,'manifest_counts':manifest_counts,
                'model_signature':signature(),'asset_hashes':assets,'code_hashes':code_hashes,
                'application_root':str(source),'preparation':json.loads((r/'preparation.json').read_text())}))"""
    snapshot = json.loads(
        cluster.call(
            "exec",
            "pod/" + selected["vessel-1"][1]["metadata"]["name"],
            "--",
            "/opt/cpython/bin/python3",
            "-c",
            code,
            source=True,
            timeout=120,
        )
    )
    for name, (generation, pod) in selected.items():
        current_deployment, current = source_pod(cluster, name)
        if (
            current["metadata"]["uid"] != pod["metadata"]["uid"]
            or current["spec"] != pod["spec"]
            or current_deployment["metadata"]["generation"] != generation
            or current["status"]["containerStatuses"][0]["imageID"] != images[name]["image_id"]
        ):
            raise RuntimeError("Source pod changed during the dataset probe: " + name)
    from harbour_fl.workload import signature

    snapshot["expected_code_hashes"] = {
        str(p.relative_to(cluster.app / "src/harbour_fl")): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (cluster.app / "src/harbour_fl").rglob("*.py")
    }
    snapshot["expected_model_signature"] = signature()
    snapshot["application_sources"] = source_identity(cluster.app)
    verify_application(snapshot, templates)
    # Bind the same host directory read-only into scaling pods. Never share the original hub PVC.
    snapshot["dataset_host_path"] = dataset_host_path(snapshot["root"], templates["vessel-1"])
    snapshot.update(
        images=images,
        node=node,
        node_hostname=node_info["metadata"]["labels"]["kubernetes.io/hostname"],
        node_info=node_info["status"]["nodeInfo"],
        allocatable=node_info["status"]["allocatable"],
    )
    # Require the metrics service now, rather than produce empty resource curves later.
    authorized(cluster, "list", "pods.metrics.k8s.io")
    authorized(cluster, "get", "nodes.metrics.k8s.io")
    metrics = json.loads(
        cluster.call(
            "get", "--raw", "/apis/metrics.k8s.io/v1beta1/namespaces/" + cluster.settings["source_namespace"] + "/pods"
        )
    )
    if not metrics.get("items"):
        raise RuntimeError("Metrics API has no source pod measurements")
    storage = cluster.get("storageclasses")["items"]
    defaults = [
        s
        for s in storage
        if s["metadata"].get("annotations", {}).get("storageclass.kubernetes.io/is-default-class") == "true"
    ]
    if len(defaults) != 1 or defaults[0].get("reclaimPolicy", "Delete") != "Delete":
        raise RuntimeError("Expected one default StorageClass with Delete reclamation for independent study storage")
    snapshot["storage_class"] = defaults[0]["metadata"]["name"]
    snapshot["source_templates"] = templates
    snapshot["identity"] = source_signature(cluster.settings, snapshot)
    if cluster.owned_namespace():
        raise RuntimeError("Scaling namespace already exists; verify cleanup before starting another attempt")
    return snapshot


def source_signature(settings, snapshot):
    """Hash effective generated workloads, retaining raw source PodSpecs only as evidence."""
    relevant = {key: value for key, value in snapshot.items() if key not in ("identity", "source_templates")}
    relevant["source_workloads"] = [
        deployment(settings, snapshot, name, role, 1, "identity")
        for name, role in (("ws-server", "hub"), ("vessel-1", "client"), ("vessel-coordinator", "coordinator"))
    ]
    return digest(relevant)


def configmap(namespace, name, files):
    """Map nested Python/CSV paths to safe ConfigMap keys and explicit volume paths."""
    data = {f"f{i}": content for i, content in enumerate(files.values())}
    if sum(len(v.encode()) for v in data.values()) > 900000:
        raise ValueError("ConfigMap exceeds the conservative payload limit")
    items = [{"key": f"f{i}", "path": path} for i, path in enumerate(files)]
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "immutable": True,
        "metadata": {"namespace": namespace, "name": name},
        "data": data,
    }, {"name": name, "configMap": {"name": name, "items": items}}


def deployment(settings, snapshot, name, role, count, job_id, replicas=1):
    """Reuse runtime binaries and modules while replacing only scaling application settings and storage."""
    source = "ws-server" if role == "hub" else ("vessel-coordinator" if role == "coordinator" else "vessel-1")
    original = snapshot["source_templates"][source]
    template = original["containers"][0]
    c = {
        k: copy.deepcopy(v)
        for k, v in template.items()
        if k in ("command", "args", "securityContext", "ports", "readinessProbe", "livenessProbe", "startupProbe")
    }
    c.update(
        name=name,
        image=snapshot["images"][source]["image"],
        imagePullPolicy="Never",
        resources=settings["resources"][role],
    )
    env = {e["name"]: e["value"] for e in template.get("env", []) if "value" in e}
    spec = {
        "containers": [c],
        "nodeSelector": {"kubernetes.io/hostname": snapshot["node_hostname"]},
        "automountServiceAccountToken": False,
        "securityContext": copy.deepcopy(original.get("securityContext", {})),
        "volumes": [],
    }
    c["volumeMounts"] = []
    if role == "hub":
        env = {k: v for k, v in env.items() if k.startswith(("MODULES_", "TLS_", "WS_", "HTTP_"))}
        if "MODULES_PATHS" not in env:
            raise ValueError("Source hub does not declare its bundled modules")
        env.update(
            STORAGE_URL="file:///app/storage",
            WS_AGENT_REGISTRY="/app/runtime/registry.yaml",
            TLS_CERT_FILE="/app/runtime/cert.pem",
            TLS_KEY_FILE="/app/runtime/key.pem",
            HTTP_WORKERS="2",
        )
        spec["volumes"].append({"name": "storage", "persistentVolumeClaim": {"claimName": "scaling-storage"}})
        c["volumeMounts"].append({"name": "storage", "mountPath": "/app/storage"})
        scratch = ("runtime", "/app/runtime")
    else:
        env = {k: v for k, v in env.items() if k in ("RUNNER_MODULE", "HARBOUR_FL_READINESS_FILE")}
        env.update(
            WS_SERVER_URL="ws://ws-server:8080/ws",
            PYO3_AGENT_ID=name,
            PYO3_PYTHONPATH="/opt/et-python",
            PYTHONPATH="/opt/et-python",
            PYTHONDONTWRITEBYTECODE="1",
            HARBOUR_FL_ROLE=role,
            HARBOUR_FL_CLIENTS=",".join(f"vessel_{i + 1}" for i in range(count)),
            HARBOUR_FL_JOB_ID=job_id,
            RECOVERY_JOB_ID=job_id,
            RECOVERY_SUPERVISOR_AGENT_ID="",
            HARBOUR_FL_MODEL_PROFILE="pretrained",
            HARBOUR_FL_DEVICE="cpu",
            HARBOUR_FL_DATA_ROOT="/scaling/data",
            HARBOUR_FL_RESUME="false",
            HARBOUR_FL_ADAPTATION="false",
            HARBOUR_FL_PROBES_ENABLED="0",
            HARBOUR_FL_DETECTOR_ENABLED="false",
            ET_BENCHMARK_GPU_FAULTS="0",
            HARBOUR_FL_IO_READ_DELAY_MS="0",
            HARBOUR_FL_AUDIO_NOISE_STD="0",
            HARBOUR_FL_IMAGE_NOISE_STD="0",
            HARBOUR_FL_MISSING_MODALITY="none",
            HARBOUR_FL_INVALID_UPDATE="false",
            HARBOUR_FL_FAIL_CHECKPOINT_WRITE_ROUND="-1",
            HARBOUR_FL_CHECKPOINT_WRITE_DELAY_SECONDS="0",
            HARBOUR_FL_LEARNING_RATE="0.001",
            HARBOUR_FL_EVALUATE_DEGRADED_INPUTS="0",
            HARBOUR_FL_AUTO_START="true",
            HARBOUR_FL_START_DELAY_SECONDS="10",
            HARBOUR_FL_ROUNDS=str(settings["rounds"]),
            HARBOUR_FL_MIN_CLIENTS=str(count),
            HARBOUR_FL_LOCAL_EPOCHS=str(settings["local_epochs"]),
            HARBOUR_FL_BATCH_SIZE=str(settings["batch_size"]),
            HARBOUR_FL_TORCH_THREADS=str(settings["torch_threads"]),
            HARBOUR_FL_TRAINING_SEED=str(settings["seed"]),
            HARBOUR_FL_ROUND_TIMEOUT_SECONDS=str(settings["round_timeout_seconds"]),
            OMP_NUM_THREADS=str(settings["torch_threads"]),
            MKL_NUM_THREADS=str(settings["torch_threads"]),
            HARBOUR_FL_MLFLOW_RUN_NAME=job_id,
        )
        if role == "client":
            env["HARBOUR_FL_CLIENT_ID"] = name.replace("-", "_")
        if not env.get("RUNNER_MODULE"):
            raise ValueError("Source runner does not declare its module")
        spec["volumes"].extend(
            [
                {"name": "live-source-data", "hostPath": {"path": snapshot["dataset_host_path"], "type": "Directory"}},
                {"name": "source-data", "emptyDir": {}},
            ]
        )
        spec["initContainers"] = [
            {
                "name": "snapshot-data",
                "image": c["image"],
                "imagePullPolicy": "Never",
                "command": [
                    "/opt/cpython/bin/python3",
                    "/snapshot-code/snapshot.py",
                    "/snapshot-source",
                    "/snapshot-output",
                    "/snapshot-code/snapshot-assets.json",
                ],
                "resources": copy.deepcopy(settings["resources"][role]),
                "securityContext": copy.deepcopy(c.get("securityContext", {})),
                "volumeMounts": [
                    {"name": "live-source-data", "mountPath": "/snapshot-source", "readOnly": True},
                    {"name": "source-data", "mountPath": "/snapshot-output"},
                    {"name": "scaling-data", "mountPath": "/snapshot-code", "readOnly": True},
                ],
            }
        ]
        c["volumeMounts"].extend(
            [
                {"name": "source-data", "mountPath": "/scaling-source", "readOnly": True},
                {"name": "scaling-data", "mountPath": "/scaling/data", "readOnly": True},
            ]
        )
        scratch = ("tmp", "/tmp")
    spec["volumes"].append({"name": scratch[0], "emptyDir": {}})
    c["volumeMounts"].append({"name": scratch[0], "mountPath": scratch[1]})
    c["env"] = [{"name": k, "value": v} for k, v in sorted(env.items())]
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "namespace": settings["namespace"]},
        "spec": {
            "replicas": replicas,
            "strategy": {"type": "Recreate"},
            "selector": {"matchLabels": {"app": name}},
            "template": {"metadata": {"labels": {"app": name, LABEL: OWNER}}, "spec": spec},
        },
    }


def documents(settings, snapshot, data, count, job_id):
    """Build a complete study namespace; no original objects appear in the apply list."""
    ns = settings["namespace"]
    data_maps, data_volume = dataset_configmaps(
        ns,
        {
            **data,
            "snapshot.py": (ROOT / "dataset_snapshot.py").read_text(),
            "snapshot-assets.json": json.dumps(snapshot.get("asset_hashes", {}), sort_keys=True),
        },
    )
    result = [
        {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": ns, "labels": {LABEL: OWNER}}},
        *data_maps,
        {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {"name": "scaling-storage", "namespace": ns},
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "storageClassName": snapshot["storage_class"],
                "resources": {"requests": {"storage": "8Gi"}},
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "ws-server", "namespace": ns},
            "spec": {"selector": {"app": "ws-server"}, "ports": [{"name": "ws", "port": 8080, "targetPort": 8080}]},
        },
        deployment(settings, snapshot, "ws-server", "hub", count, job_id),
    ]
    for name, role in [(f"vessel-{i + 1}", "client") for i in range(count)] + [("vessel-coordinator", "coordinator")]:
        doc = deployment(settings, snapshot, name, role, count, job_id, replicas=0 if role == "coordinator" else 1)
        doc["spec"]["template"]["spec"]["volumes"].append(data_volume)
        result.append(doc)
    return result


def dataset_configmaps(namespace, files):
    """Split larger workload manifests into bounded ConfigMaps projected as one dataset directory."""
    if sum(len(value.encode()) for value in files.values()) <= 800000:
        doc, volume = configmap(namespace, "scaling-data", files)
        return [doc], volume
    chunks, chunk, size = [], {}, 0
    for name, value in files.items():
        weight = len(value.encode())
        if weight > 800000:
            raise ValueError("One dataset manifest exceeds the ConfigMap payload budget")
        if chunk and size + weight > 800000:
            chunks.append(chunk)
            chunk, size = {}, 0
        chunk[name] = value
        size += weight
    if chunk:
        chunks.append(chunk)
    docs, sources = [], []
    for i, chunk in enumerate(chunks):
        doc, volume = configmap(namespace, f"scaling-data-{i}", chunk)
        docs.append(doc)
        sources.append({"configMap": volume["configMap"]})
    return docs, {"name": "scaling-data", "projected": {"sources": sources}}
