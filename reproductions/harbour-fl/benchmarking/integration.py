"""Harbour-FL application binding for the shared Kubernetes benchmark engine."""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, ClassVar

from et_experiments.backends.hub_probes import HubTaskProbeTransport
from et_experiments.benchmarking.faults import bounded_number
from et_experiments.benchmarking.probes import ServiceProbeMonitor
from et_experiments.benchmarking.probes import settings as probe_settings
from et_experiments.benchmarking.reference import fingerprint
from et_experiments.benchmarking.transfers import transfer_summary
from et_experiments.contracts import WorkloadConfig
from et_experiments.recovery.statistics import distribution
from et_experiments.tracking.mlflow import TrackingConfig
from et_experiments.utils import BenchmarkError, clean_log, iso, log_timestamp, percentile, progress
from harbour_profile import enrich_report, normalise_progress, observe_training, tracking_recovery
from workload_settings import policy_environment, policy_settings, resilience_settings

NAMESPACE = "et-harbour-fl"
CONTEXT = os.environ.get("HARBOUR_FL_KUBE_CONTEXT", "k3d-harbour-fl")
COORDINATOR = "vessel-coordinator"
CLIENT_DEPLOYMENTS = ("vessel-1", "vessel-2", "vessel-3")
SUPPORTING_DEPLOYMENTS = ("ws-server", "mlflow", "mlflow-bridge")
STRESS_IMAGE = "harbour-fl-pyo3:local"
EXPERIMENT_NAME = "vessel-multimodal-fl"
ROUND_COMPLETE = re.compile(r"coordinator completed round (\d+): (\{.*\})")
CLIENT_COMPLETE = re.compile(r"(vessel_[123]) completed round (\d+): (\{.*\})")
CHECKPOINT_WRITE = re.compile(r"checkpoint write round (\d+) completed in ([0-9.]+) seconds")


def recovery_config():
    config = json.loads((Path(__file__).parents[1] / "recovery.json").read_text())
    config["context"] = CONTEXT
    return config


def parse_logs(logs: dict[str, str], rounds_expected: int) -> dict[str, Any]:
    """Extract round progress, client metrics and checkpoint timings while retaining coordinator failure evidence."""
    coordinator = logs.get("vessel-coordinator", "")
    round_records: list[dict[str, Any]] = []
    start_at: dt.datetime | None = None
    failed = False
    for raw in coordinator.splitlines():
        line = clean_log(raw)
        timestamp = log_timestamp(line)
        if start_at is None and ("coordinator auto-start enabled" in line or "coordinator connected" in line):
            start_at = timestamp
        if "harbour FL coordinator failed" in line:
            failed = True
        match = ROUND_COMPLETE.search(line)
        if not match:
            continue
        try:
            metrics = ast.literal_eval(match.group(2))
        except (SyntaxError, ValueError):
            metrics = {}
        round_records.append(
            {
                "round": int(match.group(1)),
                "completed_at": iso(timestamp) if timestamp else None,
                "metrics": metrics,
            }
        )

    previous = start_at
    durations: list[float] = []
    for record in round_records:
        completed = (
            dt.datetime.fromisoformat(record["completed_at"].replace("Z", "+00:00")) if record["completed_at"] else None
        )
        duration = record["metrics"].get("round_work_seconds")
        if duration is None:
            duration = (completed - previous).total_seconds() if completed and previous else None
        record["duration_seconds"] = duration
        if duration is not None and duration >= 0:
            durations.append(duration)
        previous = completed or previous

    clients: dict[str, list[dict[str, Any]]] = {}
    for deployment in ("vessel-1", "vessel-2", "vessel-3"):
        for raw in logs.get(deployment, "").splitlines():
            line = clean_log(raw)
            match = CLIENT_COMPLETE.search(line)
            if not match:
                continue
            try:
                metrics = ast.literal_eval(match.group(3))
            except (SyntaxError, ValueError):
                metrics = {}
            clients.setdefault(match.group(1), []).append({"round": int(match.group(2)), "metrics": metrics})

    final_metrics = round_records[-1]["metrics"] if round_records else {}
    checkpoint_writes = [float(match.group(2)) for match in CHECKPOINT_WRITE.finditer(clean_log(coordinator))]
    return {
        "completed_rounds": len({item["round"] for item in round_records}),
        "expected_rounds": rounds_expected,
        "coordinator_failed": failed
        and not (coordinator.rfind("coordinator completed round") > coordinator.rfind("harbour FL coordinator failed")),
        "coordinator_recovered": failed
        and coordinator.rfind("coordinator completed round") > coordinator.rfind("harbour FL coordinator failed"),
        "rounds": round_records,
        "round_duration_seconds": {
            "mean": statistics.fmean(durations) if durations else None,
            "p50": percentile(durations, 0.50),
            "p95": percentile(durations, 0.95),
            "max": max(durations) if durations else None,
        },
        "client_rounds": clients,
        "checkpoint_write_seconds": checkpoint_writes,
        "final_model_metrics": final_metrics,
    }


def configure_run(backend, faults, job_id: str, scenario: dict[str, Any], started: dt.datetime) -> None:
    """Configure isolated experiment identities and faults, then start clients before the coordinator."""
    validate_compute(scenario)
    recovery = recovery_config()
    policy = policy_settings(scenario, recovery["policy"])
    backend.kubectl(
        "set", "env", f"deployment/{recovery['supervisor_deployment']}", "-n", NAMESPACE, *policy_environment(policy)
    )
    progress(started, "Starting ws-server, MLflow, and the MLflow bridge")
    for deployment in SUPPORTING_DEPLOYMENTS:
        backend.kubectl("scale", f"deployment/{deployment}", "-n", NAMESPACE, "--replicas=1")
    backend.wait_rollouts(("ws-server", "mlflow"), timeout=300, started=started)

    progress(started, "Stopping existing FL roles to prevent overlapping client identities")
    for deployment in (*CLIENT_DEPLOYMENTS, COORDINATOR):
        backend.kubectl("scale", f"deployment/{deployment}", "-n", NAMESPACE, "--replicas=0")
    for deployment in (*CLIENT_DEPLOYMENTS, COORDINATOR):
        backend.kubectl(
            "wait",
            "--for=delete",
            "pod",
            "-n",
            NAMESPACE,
            "-l",
            f"app={deployment}",
            "--timeout=120s",
            check=False,
        )

    fault = scenario.get("fault", {})
    device = str(scenario.get("device", "cpu"))
    checkpoint_delay = float(scenario.get("checkpoint_write_delay_seconds", 0))
    faults.prepare_fault(scenario.get("preparation_fault", fault), started)

    progress(started, f"Configuring three FL clients for job {job_id}")
    supervised = scenario.get("recovery_mode", "observe-only") != "disabled" and scenario.get("recovery_enabled", True)
    common = [
        f"HARBOUR_FL_JOB_ID={job_id}",
        f"RECOVERY_JOB_ID={job_id}",
        f"RECOVERY_SUPERVISOR_AGENT_ID={'recovery-supervisor' if supervised else ''}",
        f"HARBOUR_FL_LOCAL_EPOCHS={int(scenario.get('local_epochs', 1))}",
        f"HARBOUR_FL_BATCH_SIZE={int(scenario.get('batch_size', 16))}",
        f"HARBOUR_FL_TORCH_THREADS={int(scenario.get('torch_threads', 2))}",
        f"HARBOUR_FL_TRAINING_SEED={int(scenario.get('training_seed', 17))}",
        f"HARBOUR_FL_MODEL_PROFILE={scenario.get('model_profile', 'pretrained')}",
        f"HARBOUR_FL_EVALUATE_DEGRADED_INPUTS={int(bool(scenario.get('evaluate_degraded_inputs', False)))}",
    ]
    probes = probe_settings(scenario, len(CLIENT_DEPLOYMENTS))
    from harbour_fl.service_checks import observer_id

    common.extend(
        [
            f"HARBOUR_FL_PROBES_ENABLED={int(probes['enabled'])}",
            f"HARBOUR_FL_PROBE_OBSERVER_ID={observer_id(job_id)}",
            f"HARBOUR_FL_PROBE_INTERVAL_SECONDS={probes.get('interval_seconds', 2)}",
            f"HARBOUR_FL_PROBE_TIMEOUT_SECONDS={probes.get('timeout_seconds', 5)}",
            f"HARBOUR_FL_PROBE_PAYLOAD_BYTES={probes.get('rtt_payload_bytes', 32)}",
        ]
    )
    for deployment in (*CLIENT_DEPLOYMENTS, COORDINATOR):
        env = list(common)
        selected_device = scenario.get("client_devices", {}).get(deployment, device)
        env.append(f"HARBOUR_FL_DEVICE={selected_device if deployment in CLIENT_DEPLOYMENTS else 'cpu'}")
        enabled = fault.get("type") == "gpu_failure" and deployment == str(fault.get("target"))
        env.append(f"ET_BENCHMARK_GPU_FAULTS={int(enabled)}")
        env.append(f"HARBOUR_FL_ADAPTATION={'true' if scenario.get('adaptation') else 'false'}")
        if deployment == "vessel-coordinator":
            env.extend(
                (
                    "HARBOUR_FL_AUTO_START=true",
                    "HARBOUR_FL_START_DELAY_SECONDS=20",
                    f"HARBOUR_FL_ROUNDS={int(scenario['rounds'])}",
                    f"HARBOUR_FL_ROUND_TIMEOUT_SECONDS={int(scenario.get('round_timeout_seconds', 180))}",
                    f"HARBOUR_FL_MLFLOW_RUN_NAME={job_id}",
                    f"HARBOUR_FL_MIN_CLIENTS={int(scenario.get('minimum_clients', 3))}",
                    f"HARBOUR_FL_CHECKPOINT_INTERVAL={int(scenario.get('checkpoint_interval', 1))}",
                    f"HARBOUR_FL_CHECKPOINT_WRITE_DELAY_SECONDS={checkpoint_delay}",
                )
            )
            if fault.get("type") == "checkpoint_write_failure":
                env.append(f"HARBOUR_FL_FAIL_CHECKPOINT_WRITE_ROUND={int(fault.get('round', 1))}")
        if deployment == str(fault.get("target")) or (
            deployment == "vessel-coordinator" and scenario.get("adaptation")
        ):
            if fault.get("type") == "dataset_read_latency":
                delay_ms = bounded_number(fault, "delay_ms", 0, 5000, 25)
                env.append(f"HARBOUR_FL_IO_READ_DELAY_MS={delay_ms:g}")
            elif fault.get("type") == "audio_noise":
                snr = audio_snr(fault)
                env.append(f"HARBOUR_FL_AUDIO_SNR_DB={snr:g}")
            elif fault.get("type") == "image_noise":
                noise_std = image_noise(fault)
                env.append(f"HARBOUR_FL_IMAGE_NOISE_STD={noise_std:g}")
            elif fault.get("type") == "missing_modality":
                modality = str(fault.get("modality", ""))
                if modality not in {"audio", "image"}:
                    raise BenchmarkError(f"unsupported missing modality: {modality}")
                env.append(f"HARBOUR_FL_MISSING_MODALITY={modality}")
            elif fault.get("type") == "invalid_update":
                env.append("HARBOUR_FL_INVALID_UPDATE=true")
        backend.kubectl("set", "env", f"deployment/{deployment}", "-n", NAMESPACE, *env)

    progress(started, "Starting the three clients")
    for deployment in ("vessel-1", "vessel-2", "vessel-3"):
        backend.kubectl("scale", f"deployment/{deployment}", "-n", NAMESPACE, "--replicas=1")
    backend.wait_rollouts(("vessel-1", "vessel-2", "vessel-3"), started=started)
    time.sleep(5)

    progress(started, "Starting the coordinator after all clients")
    backend.kubectl("scale", "deployment/vessel-coordinator", "-n", NAMESPACE, "--replicas=1")
    backend.wait_rollouts(("vessel-coordinator",), started=started)
    # The bridge may initially restart while its two services come up. Do not
    # start the measurement window until it has settled.
    backend.wait_rollouts(("mlflow-bridge",), timeout=300, started=started)
    progress(started, "All benchmark workloads are ready; coordinator auto-start is armed")


def restore(backend) -> None:
    """Restore manual application settings for every workload and report all failed restoration commands."""
    errors = []

    def attempt(*arguments):
        """Attempt one checked Kubernetes command without preventing the remaining restoration steps."""
        try:
            backend.kubectl(*arguments, check=True)
        except (BenchmarkError, OSError, subprocess.TimeoutExpired) as error:
            errors.append(f"{' '.join(arguments[:3])}: {error}")

    recovery = recovery_config()
    manual_job = recovery["job_id"]
    attempt(
        "set",
        "env",
        f"deployment/{recovery['supervisor_deployment']}",
        "-n",
        NAMESPACE,
        *policy_environment(recovery["policy"]),
    )
    for deployment in (*CLIENT_DEPLOYMENTS, COORDINATOR):
        keys = [
            f"RECOVERY_JOB_ID={manual_job}",
            f"HARBOUR_FL_JOB_ID={manual_job}",
            "RECOVERY_SUPERVISOR_AGENT_ID=recovery-supervisor",
            "HARBOUR_FL_INVALID_UPDATE-",
            "HARBOUR_FL_MIN_CLIENTS-",
            "HARBOUR_FL_ADAPTATION-",
            "HARBOUR_FL_IO_READ_DELAY_MS-",
            "HARBOUR_FL_AUDIO_NOISE_STD-",
            "HARBOUR_FL_AUDIO_SNR_DB-",
            "HARBOUR_FL_IMAGE_NOISE_STD-",
            "HARBOUR_FL_MISSING_MODALITY-",
            "HARBOUR_FL_DEVICE-",
            "ET_BENCHMARK_GPU_FAULTS-",
            "HARBOUR_FL_LOCAL_EPOCHS-",
            "HARBOUR_FL_BATCH_SIZE-",
            "HARBOUR_FL_TORCH_THREADS-",
            "HARBOUR_FL_TRAINING_SEED-",
            "HARBOUR_FL_MODEL_PROFILE-",
            "HARBOUR_FL_EVALUATE_DEGRADED_INPUTS-",
            "HARBOUR_FL_CHECKPOINT_INTERVAL-",
            "HARBOUR_FL_CHECKPOINT_WRITE_DELAY_SECONDS-",
            "HARBOUR_FL_PROBES_ENABLED-",
            "HARBOUR_FL_PROBE_OBSERVER_ID-",
            "HARBOUR_FL_PROBE_INTERVAL_SECONDS-",
            "HARBOUR_FL_PROBE_TIMEOUT_SECONDS-",
            "HARBOUR_FL_PROBE_PAYLOAD_BYTES-",
        ]
        if deployment == "vessel-coordinator":
            keys.extend(
                (
                    "HARBOUR_FL_AUTO_START-",
                    "HARBOUR_FL_START_DELAY_SECONDS-",
                    "HARBOUR_FL_ROUNDS-",
                    "HARBOUR_FL_ROUND_TIMEOUT_SECONDS-",
                    "HARBOUR_FL_MLFLOW_RUN_NAME-",
                    "HARBOUR_FL_FAIL_CHECKPOINT_WRITE_ROUND-",
                )
            )
        else:
            keys.extend(
                (
                    "HARBOUR_FL_IO_READ_DELAY_MS-",
                    "HARBOUR_FL_AUDIO_NOISE_STD-",
                    "HARBOUR_FL_AUDIO_SNR_DB-",
                    "HARBOUR_FL_IMAGE_NOISE_STD-",
                    "HARBOUR_FL_MISSING_MODALITY-",
                )
            )
        attempt("set", "env", f"deployment/{deployment}", "-n", NAMESPACE, *keys)
        attempt("scale", f"deployment/{deployment}", "-n", NAMESPACE, "--replicas=1")
    if errors:
        raise BenchmarkError("Application restoration failed: " + "; ".join(errors))


def audio_snr(fault):
    """Validate a cached waveform-SNR condition and reject the incompatible log-mel noise setting."""
    if "noise_std" in fault:
        raise BenchmarkError("audio_noise requires snr_db; log-mel noise_std is not supported")
    value = fault.get("snr_db", 10)
    if type(value) not in (int, float) or value not in (0, 10, 20):
        raise BenchmarkError("audio snr_db must be a prepared condition: 0, 10 or 20")
    return value


def image_noise(fault):
    """Accept only image-noise levels represented in the prepared feature archives."""
    value = fault.get("noise_std", 0.2)
    if type(value) not in (int, float) or value not in (0.05, 0.1, 0.2):
        raise BenchmarkError("image noise_std must be a prepared condition: 0.05, 0.10 or 0.20")
    return value


def application_fault(fault: dict[str, Any]) -> dict[str, Any] | None:
    """Validate application-level fault settings and describe their scope for the shared benchmark engine."""
    kind = fault.get("type")
    if kind == "checkpoint_write_failure":
        return {"status": "configured", "round": int(fault.get("round", 1))}
    if kind not in {"dataset_read_latency", "audio_noise", "image_noise", "missing_modality", "invalid_update"}:
        return None
    if fault.get("target") not in CLIENT_DEPLOYMENTS:
        raise BenchmarkError("data-quality faults require an allowlisted client target")
    result = {"status": "configured", "scope": "client-training-and-validation"}
    if kind == "dataset_read_latency":
        result.update(delay_ms=bounded_number(fault, "delay_ms", 0, 5000, 25), scope="prepared-feature-archive-reads")
    elif kind == "missing_modality":
        modality = fault.get("modality")
        if modality not in {"audio", "image"}:
            raise BenchmarkError(f"unsupported missing modality: {modality}")
        result["modality"] = modality
    elif kind == "audio_noise":
        result["snr_db"] = audio_snr(fault)
    elif kind == "image_noise":
        result["noise_std"] = image_noise(fault)
    return result


def validate_compute(scenario):
    """Reject unsupported execution modes before making any Kubernetes deployment changes."""
    device = str(scenario.get("device", "cpu"))
    clients = scenario.get("client_devices", {})
    if not isinstance(clients, dict) or not set(clients).issubset(CLIENT_DEPLOYMENTS):
        raise BenchmarkError("client_devices must map allowlisted clients to execution devices")
    selected = [device, *clients.values()]
    if any(value not in {"cpu", "cuda", "mps", "auto"} for value in selected):
        raise BenchmarkError(f"unsupported compute device: {device}")
    fault = scenario.get("fault", {})
    target_device = clients.get(fault.get("target"), device)
    if fault.get("type") == "gpu_failure" and target_device not in {"cuda", "mps"}:
        raise BenchmarkError("GPU scenarios require an explicit cuda or mps device; CPU fallback is disabled")
    if "mps" in selected:
        raise BenchmarkError("MPS requires native macOS execution; Kubernetes Linux containers cannot expose Apple MPS")


class HarbourFLAdapter:
    config = WorkloadConfig(
        CONTEXT,
        NAMESPACE,
        COORDINATOR,
        CLIENT_DEPLOYMENTS,
        (*SUPPORTING_DEPLOYMENTS, "recovery-supervisor"),
        STRESS_IMAGE,
    )
    tracking = TrackingConfig(
        EXPERIMENT_NAME,
        (
            "global_test_accuracy",
            "global_test_loss",
            "global_test_macro_f1",
            "mean_client_validation_accuracy",
            "participating_clients",
            "round_work_seconds",
        ),
    )
    metric_aliases: ClassVar[dict[str, str]] = {
        "accuracy": "global_test_accuracy",
        "macro_f1": "global_test_macro_f1",
        "loss": "global_test_loss",
        "participation": "participating_clients",
    }

    def recovery_config(self):
        return recovery_config()

    def parse_logs(self, logs, rounds_expected):
        return parse_logs(logs, rounds_expected)

    def expected_units(self, scenario):
        return int(scenario["rounds"])

    def prepare_scenario(self, scenario):
        """Validate workload controls before deployment changes and translate FL settings into generic engine
        controls."""
        validate_compute(scenario)
        policy_settings(scenario, recovery_config()["policy"])
        for key, upper in (
            ("minimum_clients", 3),
            ("local_epochs", 50),
            ("batch_size", 256),
            ("torch_threads", 32),
            ("training_seed", 2**31 - 1),
            ("checkpoint_interval", 100),
        ):
            value = scenario.get(
                key, {"minimum_clients": 3, "batch_size": 16, "torch_threads": 2, "training_seed": 17}.get(key, 1)
            )
            if type(value) is not int or not 1 <= value <= upper:
                raise BenchmarkError(f"{key} must be an integer between 1 and {upper}")
        delay = scenario.get("checkpoint_write_delay_seconds", 0)
        if type(delay) not in (int, float) or not 0 <= delay <= 30:
            raise BenchmarkError("checkpoint_write_delay_seconds must be between 0 and 30")
        scenario = {**scenario, "model_profile": "pretrained", "fault": dict(scenario.get("fault", {"type": "none"}))}
        scenario["recovery_resilience"] = resilience_settings(scenario, recovery_config().get("resilience", {}))
        scenario["service_probes"] = probe_settings(scenario, len(self.config.workers))
        if type(scenario.get("evaluate_degraded_inputs", False)) is not bool:
            raise BenchmarkError("evaluate_degraded_inputs must be boolean")
        if scenario.get("evaluate_degraded_inputs"):
            scenario["completion_event"] = "completed"
        fault = scenario["fault"]
        application_fault(fault)
        if fault.get("type") == "client_oom":
            fault["type"] = "worker_oom"
        if "after_round" in fault:
            fault["after_work_unit"] = fault.pop("after_round")
        return scenario

    def observe(self, logs, tracking_state, expected_units, started):
        return normalise_progress(
            observe_training(parse_logs(logs, expected_units), tracking_state, expected_units, started)
        )

    def enrich_report(self, report):
        """Attach application measurements and observed model identities, failing reports with missing workload
        evidence."""
        profiles = {}
        workload_profiles = {}
        for event in report.get("diagnostic_events", []):
            if event.get("participant") in {"vessel_1", "vessel_2", "vessel_3"}:
                event["participant_workload"] = event["participant"].replace("_", "-")
            if event["kind"] == "image-preprocessing" and isinstance(event.get("settings"), dict):
                profiles.setdefault(event["workload"], {})[json.dumps(event["settings"], sort_keys=True)] = event[
                    "settings"
                ]
            if event["kind"] == "workload-profile" and isinstance(event.get("settings"), dict):
                workload_profiles.setdefault(event["workload"], {})[json.dumps(event["settings"], sort_keys=True)] = (
                    event["settings"]
                )
        if workload_profiles:
            observed_workloads = {
                name: [settings[key] for key in sorted(settings)]
                for name, settings in sorted(workload_profiles.items())
            }
            report["observed_workloads"] = observed_workloads
            if "reference" in report:
                report["reference"]["conditions"]["observed_workloads"] = observed_workloads
        if profiles:
            observed = {
                name: [settings[key] for key in sorted(settings)] for name, settings in sorted(profiles.items())
            }
            report["image_preprocessing"] = observed
            if "reference" in report:
                report["reference"]["conditions"]["image_preprocessing"] = observed
                report["reference"]["fingerprint"] = fingerprint(report["reference"]["conditions"])
        elif workload_profiles and "reference" in report:
            report["reference"]["fingerprint"] = fingerprint(report["reference"]["conditions"])
        if report.get("effective_scenario", {}).get("model_profile") == "pretrained":
            missing = sorted(set(self.config.workloads) - set(workload_profiles))
            if missing:
                report["status"] = "failed"
                report["workload_evidence_missing"] = missing
                if "reference" in report:
                    report["reference"].setdefault("missing", []).append("observed-workloads")
        slis = enrich_report(report, self.metric_aliases, len(self.config.workers))
        report["transfer_measurements"] = transfer_summary(report.get("diagnostic_events", []), self.config.workers)
        report["prediction_input_evaluations"] = [
            event
            for event in report.get("diagnostic_events", [])
            if event["kind"] == "quality-evaluation" and event["workload"] == self.config.primary
        ]
        probes = report.get("service_probes", {})
        if probes.get("settings", {}).get("enabled"):
            slis["functional_availability_ratio"] = probes.get("availability_ratio")
            slis["functional_observation_coverage"] = probes.get("observation_coverage")
            report["communication_probes"] = communication_summary(report.get("diagnostic_events", []))
        return slis

    def service_probe_monitor(self, backend, job_id, scenario):
        """Check probe support in each runner before creating the independent functional-check observer."""
        from harbour_fl.service_checks import REQUEST, RESULT, observer_id

        for deployment in (*CLIENT_DEPLOYMENTS, COORDINATOR):
            result = backend.kubectl(
                "exec",
                f"deployment/{deployment}",
                "-n",
                NAMESPACE,
                "-c",
                deployment,
                "--",
                "/opt/cpython/bin/python3",
                "-c",
                "import websocket; from harbour_fl.service_checks import VERSION; assert VERSION == 1",
                check=False,
                timeout=15,
            )
            if result.returncode:
                diagnostic = (result.stderr or result.stdout or "no command output").strip()
                raise BenchmarkError(f"{deployment} service-probe check failed: {diagnostic}")
        transport = HubTaskProbeTransport(backend, job_id, observer_id(job_id), COORDINATOR, REQUEST, RESULT)
        return ServiceProbeMonitor(transport, self.config.workers, scenario["service_probes"])

    def reference_conditions(self, scenario):
        """Capture source, manifest, training and resource identities for matched experiment comparisons."""
        root = Path(__file__).parents[1]
        data = root / "data/croatia_1g_logmel_images"
        paths = [data / "test_manifest.csv", data / "class_taxonomy.json"]
        paths.extend(
            data / "clients" / f"vessel_{i}" / name
            for i in range(1, 4)
            for name in ("train_manifest.csv", "val_manifest.csv")
        )
        manifests = {
            str(p.relative_to(data)): hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None for p in paths
        }
        source = hashlib.sha256()
        for path in sorted((root / "src/harbour_fl").rglob("*.py")):
            source.update(str(path.relative_to(root / "src/harbour_fl")).encode())
            source.update(path.read_bytes())
        return {
            "profile": "three-client-multimodal-classification",
            "clients": list(CLIENT_DEPLOYMENTS),
            "dataset_manifests": manifests,
            "missing_conditions": ["dataset-manifests"] if any(v is None for v in manifests.values()) else [],
            "application_source_sha256": source.hexdigest(),
            "modalities": ["audio-embedding", "image-embedding"],
            "encoder_execution": "offline-frozen-feature-preparation",
            "aggregation": "fedavg",
            "model_profile": scenario.get("model_profile", "pretrained"),
            "rounds": scenario["rounds"],
            "local_epochs": scenario.get("local_epochs", 1),
            "batch_size": scenario.get("batch_size", 16),
            "torch_threads": scenario.get("torch_threads", 2),
            "training_seed": scenario.get("training_seed", 17),
            "minimum_clients": scenario.get("minimum_clients", 3),
            "adaptation": bool(scenario.get("adaptation")),
            "device": scenario.get("device", "cpu"),
            "client_devices": scenario.get("client_devices", {}),
            "checkpoint_interval": scenario.get("checkpoint_interval", 1),
            "recovery_policy": policy_settings(scenario, recovery_config()["policy"]),
            "recovery_resilience": resilience_settings(scenario, recovery_config().get("resilience", {})),
            "checkpoint_write_delay_seconds": scenario.get("checkpoint_write_delay_seconds", 0),
            "round_timeout_seconds": scenario.get("round_timeout_seconds", 180),
            "service_probes": probe_settings(scenario, len(self.config.workers)),
            "evaluate_degraded_inputs": bool(scenario.get("evaluate_degraded_inputs", False)),
        }

    def tracking_recovery(self, state, expected_units, fault, verified_at=None):
        return tracking_recovery(state, expected_units, self.metric_aliases["loss"], fault, verified_at)

    def configure_run(self, backend, faults, job_id, scenario, started):
        configure_run(backend, faults, job_id, scenario, started)

    def restore(self, backend):
        restore(backend)

    def application_fault(self, fault):
        return application_fault(fault)


def communication_summary(events):
    """Summarize observed client RTT attempts; missing client periods are not availability evidence."""
    result = {}
    for worker in CLIENT_DEPLOYMENTS:
        rows = [e for e in events if e.get("workload") == worker and e.get("kind") == "communication-probe"]
        errors = [
            e for e in events if e.get("workload") == worker and e.get("kind") == "communication-probe-observer-error"
        ]
        successes = [e for e in rows if e.get("status") == "success"]
        result[worker] = {
            "status": "observed" if rows else "not-observed",
            "attempts": len(rows),
            "successful": len(successes),
            "timeouts": sum(e.get("status") == "timeout" for e in rows),
            "failures": sum(e.get("status") == "failure" for e in rows),
            "observer_errors": len(errors),
            "success_ratio": len(successes) / len(rows) if rows else None,
            "rtt_ms": distribution([e.get("rtt_ms") for e in successes]),
            "payload_bytes": sorted({e["payload_bytes"] for e in rows if "payload_bytes" in e}),
            "first_observed_at": min((e["at"] for e in rows), default=None),
            "last_observed_at": max((e["at"] for e in rows), default=None),
            "scope": "client-to-hub WebSocket Ping/Pong; connection setup excluded",
            "coverage_note": "only emitted attempts; client downtime and missing logs are unobserved",
        }
    return result
