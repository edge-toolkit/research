"""Isolated Linux per-vessel network conditions for a Core-compatible 25,000-agent heartbeat fleet."""

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import platform
import re
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path

from network import RETRY_IPS, SERVICES, setup, topology

ROOT = Path(__file__).resolve().parent
SOURCES = [
    "study.py",
    "tc_setup.py",
    "network.py",
    "diagnostics.py",
    "monitor.py",
    "settings.json",
    "Dockerfile",
    ".dockerignore",
    ".gitignore",
    "README.md",
    "mise.toml",
    "mise.lock",
    "pyproject.toml",
    "uv.lock",
    "driver/Cargo.toml",
    "driver/Cargo.lock",
    "driver/src/main.rs",
    "tests/test_study.py",
    "tests/test_safety.py",
]
BUILD_INPUTS = [
    "Dockerfile",
    ".dockerignore",
    "tc_setup.py",
    "monitor.py",
    "diagnostics.py",
    "driver/Cargo.toml",
    "driver/Cargo.lock",
    "driver/src/main.rs",
]


def build_identity():
    return captured_build_identity({name: digest(ROOT / name) for name in BUILD_INPUTS})


def captured_build_identity(sources):
    return hashlib.sha256(
        json.dumps({name: sources[name] for name in BUILD_INPUTS}, sort_keys=True).encode()
    ).hexdigest()


def command(args, timeout=120, check=True):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    if check and result.returncode:
        raise RuntimeError(f"{' '.join(map(str, args))}: {result.stderr.strip() or result.stdout.strip()}")
    return result


def save(path, document):
    """Flush journal replacements before startup can create Docker resources."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_mkdir(path):
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for folder in reversed(missing):
        folder.mkdir(exist_ok=True)
        sync_directory(folder.parent)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compose(project, file, *args, **kwargs):
    return command(["docker", "compose", "-p", project, "-f", str(file), *args], **kwargs)


def prepare():
    runtime = ROOT / ".runtime"
    runtime.mkdir(exist_ok=True)
    core_tag = "et-ws-server-core:local"
    core = json.loads(command(["docker", "image", "inspect", core_tag]).stdout)[0]
    metadata = {"tag": core_tag, "id": core["Id"], "architecture": core["Architecture"]}
    inputs = build_identity()
    image = f"et-hub-network-tools:{core['Id'].split(':')[-1][:12]}"
    subprocess.run(["docker", "build", "--build-arg", f"CORE_IMAGE={core_tag}", "-t", image, str(ROOT)], check=True)
    current = json.loads(command(["docker", "image", "inspect", core_tag]).stdout)[0]
    if current["Id"] != core["Id"]:
        raise RuntimeError("Core image changed during helper preparation; run prepare again")
    if inputs != build_identity():
        raise RuntimeError("Helper source changed during preparation; run prepare again")
    info = json.loads(command(["docker", "image", "inspect", image]).stdout)[0]
    save(
        runtime / "image.json",
        {
            "tag": image,
            "id": info["Id"],
            "architecture": info["Architecture"],
            "core": metadata,
            "build_inputs": inputs,
            "compiler": command(["docker", "run", "--rm", image, "cat", "/build-compiler.txt"]).stdout.strip(),
        },
    )


def image_identity():
    path = ROOT / ".runtime/image.json"
    if not path.exists():
        raise RuntimeError("Run prepare first")
    image = json.loads(path.read_text())
    if image.get("build_inputs") != build_identity():
        raise RuntimeError("Helper build inputs changed; run prepare again")
    actual = command(["docker", "image", "inspect", image["tag"], "--format", "{{.Id}}"]).stdout.strip()
    if actual != image["id"]:
        raise RuntimeError("Image identity changed; run prepare again")
    core = image["core"]
    current = command(["docker", "image", "inspect", core["tag"], "--format", "{{.Id}}"]).stdout.strip()
    if current != core["id"]:
        raise RuntimeError("Shared Core release changed; run prepare again")
    return image


def owned_resources(project):
    """Require all owned Docker resource inventories to be empty."""
    for kind, args in (
        ("containers", ["ps", "-aq"]),
        ("networks", ["network", "ls", "-q"]),
        ("volumes", ["volume", "ls", "-q"]),
    ):
        result = command(["docker", *args, "--filter", f"label=com.docker.compose.project={project}"])
        if result.stdout.strip():
            raise RuntimeError("Owned " + kind + " remain: " + project)


def verify_daemon(daemon):
    if command(["docker", "info", "--format", "{{.ID}}"]).stdout.strip() != daemon:
        raise RuntimeError("Docker daemon differs from startup; cleanup remains unverified")


class ProtectionWatch:
    """Poll host protection independently of optional, potentially slow Docker metrics."""

    def __init__(self, sample):
        self.sample = sample
        self.stopped = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.run, name="host-protection")

    def run(self):
        while not self.stopped.wait(2):
            try:
                _, reason = self.sample()
                if reason:
                    return
            except Exception as exc:  # noqa: BLE001 - propagate guard failure after it stops the workload
                self.error = exc
                return

    def close(self):
        self.stopped.set()
        if self.thread.ident is not None:
            self.thread.join()


def observe_cohort(directory):
    """Request a new driver observation rather than reuse a periodic log row."""
    request = uuid.uuid4().hex
    save(directory / "cohort-request.json", {"request": request})
    deadline = time.monotonic() + 5
    path = directory / "cohort-observation.json"
    while time.monotonic() < deadline:
        if path.is_file():
            response = json.loads(path.read_text())
            if response.get("request") == request:
                return response["cohort"]
        time.sleep(0.05)
    raise RuntimeError("Fresh cohort observation unavailable; impairment gate withheld")


@contextlib.contextmanager
def output_lock(output):
    """Coordinate runs and cleanup across checkouts using the resolved evidence directory."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def workload_host(image):
    """Fail closed when the Docker host counters cannot be read."""
    return json.loads(
        command(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                image["id"],
                "python3",
                "/usr/local/lib/hub-network/monitor.py",
                "--host",
            ],
            timeout=15,
        ).stdout
    )


class HostGuard:
    def __init__(self, config, initial):
        self.config, self.previous, self.initial = config, initial, initial
        self.busy = 0
        self.began = time.monotonic()

    def sample(self, current):
        if current["boot_id"] != self.initial["boot_id"]:
            raise RuntimeError("Docker host changed during the attempt")
        total = current["cpu_total"] - self.previous["cpu_total"]
        idle = current["cpu_idle"] - self.previous["cpu_idle"]
        cpu = 100 * (1 - idle / total) if total > 0 else 0
        self.previous = current
        self.busy = self.busy + 1 if cpu > self.config["maximum_host_cpu_percent"] else 0
        sample = {
            **current,
            "host_cpu_percent": cpu,
            "swap_out_growth_bytes": max(0, current["swap_out_bytes"] - self.initial["swap_out_bytes"]),
        }
        guard = None
        if current["available_memory_bytes"] < self.config["minimum_available_memory_bytes"]:
            guard = "host-memory-reserve"
        elif sample["swap_out_growth_bytes"] > self.config["maximum_swap_out_growth_bytes"]:
            guard = "host-swap-growth"
        elif self.busy >= self.config["host_cpu_guard_samples"]:
            guard = "host-cpu-budget"
        elif time.monotonic() - self.began > (
            self.config["registration_budget_seconds"]
            + self.config["hold_seconds"]
            + self.config.get("settle_seconds", 0)
            + 2100
        ):
            guard = "whole-job-budget"
        return sample, guard


def cleanup_attempts(output):
    """Retry teardown only for attempts recorded on this Docker daemon."""
    daemon = command(["docker", "info", "--format", "{{.ID}}"]).stdout.strip()
    if not daemon:
        raise RuntimeError("Cannot identify the Docker daemon")
    attempts = []
    for folder in sorted(output.glob("*/attempt-*")):
        report_path = folder / "report.json"
        if report_path.is_file() and json.loads(report_path.read_text()).get("cleanup_verified") is True:
            continue
        path = folder / "project.json"
        if not path.is_file():
            raise RuntimeError("Cleanup target is unknown: " + str(folder))
        target = json.loads(path.read_text())
        if target.get("phase") == "not-started":
            report = {"outcome": "interrupted-before-start", "cleanup_verified": True}
            save(report_path, report)
            continue
        if target.get("daemon_id") != daemon or not re.fullmatch(
            r"et-hub-network-[0-9a-f]{10}", target.get("project", "")
        ):
            raise RuntimeError("Cleanup target differs or is invalid: " + str(folder))
        attempts.append((folder, target["project"]))
    for folder, project in attempts:
        report_path = folder / "report.json"
        report = json.loads(report_path.read_text()) if report_path.is_file() else {"outcome": "interrupted"}
        errors = []
        try:
            verify_daemon(daemon)
            try:
                down = compose(
                    project,
                    folder / "compose.json",
                    "down",
                    "--volumes",
                    "--remove-orphans",
                    check=False,
                )
                if down.returncode:
                    report.setdefault("cleanup_diagnostics", []).append(down.stderr)
            except subprocess.TimeoutExpired as exc:
                report.setdefault("cleanup_diagnostics", []).append(str(exc))
            owned_resources(project)
            verify_daemon(daemon)
        except Exception as exc:  # noqa: BLE001 - preserve the failed cleanup verification for a later retry
            errors.append(str(exc))
        report.update(
            cleanup_verified=not errors,
            cleanup_errors=errors,
            cleanup_actor="explicit network cleanup command",
        )
        save(report_path, report)
        if errors:
            raise RuntimeError("Cleanup remains unverified: " + "; ".join(errors))


def selected_cases(config, case_id):
    """Always qualify exactly one baseline before any selected constraint."""
    cases = config["cases"]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Every configured case ID must be unique")
    baseline = [case for case in cases if case["id"] == "baseline"]
    if len(baseline) != 1:
        raise ValueError("Exactly one baseline case is required")
    if case_id and not any(case["id"] == case_id for case in cases):
        raise ValueError("Unknown case")
    return baseline + [case for case in cases if case["id"] != "baseline" and (not case_id or case["id"] == case_id)]


def json_lines(path):
    result = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return result


def execute(case, config, image, directory):
    config = {**config, **case.get("observation", {})}
    daemon = command(["docker", "info", "--format", "{{.ID}}"]).stdout.strip()
    if not daemon:
        raise RuntimeError("Cannot identify the Docker daemon")
    durable_mkdir(directory.parent)
    with tempfile.TemporaryDirectory(dir=directory.parent, prefix=".attempt-") as staging:
        published = Path(staging) / "published"
        save(published / "project.json", {"phase": "not-started", "daemon_id": daemon})
        published.rename(directory)
        sync_directory(directory.parent)
    project = "et-hub-network-" + uuid.uuid4().hex[:10]
    file = directory / "compose.json"
    effective = {
        **config,
        "tolerant_timeouts": case.get("tolerant_timeouts", False),
        "reconnect": case.get("reconnect", False),
    }
    effective["recovery_policy"] = {**config.get("recovery_policy", {}), **case.get("policy", {})}
    save(file, topology(image["id"], directory, effective))
    report = {
        "case": case,
        "project": project,
        "outcome": "not-established",
        "resource_samples": [],
        "cleanup_verified": False,
    }
    save(directory / "report.json", report)
    process = None
    began = time.monotonic()
    log = None
    started = False
    watcher = None
    try:
        host_before_start = workload_host(image)
        if host_before_start["available_memory_bytes"] < config.get("minimum_available_memory_bytes", 0):
            report["outcome"] = "protection:host-memory-reserve"
            report["resource_samples"].append({"phase": "before-start", **host_before_start})
            return report
        save(directory / "project.json", {"project": project, "daemon_id": daemon, "phase": "starting"})
        started = True
        compose(project, file, "up", "-d", timeout=120)
        ids = {side: compose(project, file, "ps", "-q", side).stdout.strip() for side in SERVICES}
        for _ in range(30):
            ready = compose(
                project,
                file,
                "exec",
                "-T",
                "hub",
                "python3",
                "-c",
                "import socket; socket.create_connection(('127.0.0.1',8080),timeout=1,"
                "source_address=('127.0.0.1',50000)).close()",
                check=False,
            )
            if ready.returncode == 0:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("Core listener did not become ready before traffic shaping")
        print("  Configuring dedicated rate and impairment routers", flush=True)
        invoke = lambda *args, **kwargs: compose(project, file, *args, **kwargs)
        report["accounting_preflight"] = {
            side: json.loads(invoke("exec", "-T", side, "python3", "/usr/local/lib/hub-network/monitor.py").stdout)
            for side in SERVICES
        }

        def host():
            return json.loads(
                invoke(
                    "exec", "-T", "driver", "python3", "/usr/local/lib/hub-network/monitor.py", "--host", timeout=15
                ).stdout
            )

        guard = HostGuard(config, host())
        guard_lock = threading.Lock()

        def stop_workload():
            if process is not None:
                try:
                    process.terminate()
                except OSError as exc:
                    report.setdefault("protection_stop_errors", []).append(str(exc))
            try:
                compose(project, file, "stop", "--timeout", "1", "driver", "hub", timeout=10)
            except Exception as exc:  # noqa: BLE001 - preserve protection evidence if stopping also fails
                report.setdefault("protection_stop_errors", []).append(str(exc))

        def protect_unlocked(phase, latest):
            try:
                sample, reason = guard.sample(host())
            except Exception as exc:
                report["host_sampling_error"] = f"{type(exc).__name__}: {exc}"
                stop_workload()
                raise
            sample.update(at=time.time(), phase=phase, driver=latest)
            report["resource_samples"].append(sample)
            if reason:
                report["outcome"] = "protection:" + reason
                stop_workload()
            return sample, reason

        def protect(phase, latest):
            with guard_lock:
                if report["outcome"].startswith("protection:"):
                    return report["resource_samples"][-1], report["outcome"].removeprefix("protection:")
                return protect_unlocked(phase, latest)

        _, reason = protect("setup", {})
        if reason:
            return report
        network = setup(
            invoke,
            lambda identifier: command(["docker", "inspect", identifier]).stdout,
            ids,
            {**case, "loss_percent": 0} if case.get("loss_after_registration") else case,
            effective,
            directory,
        )
        report["policy"] = {
            "tolerant_timeouts": effective["tolerant_timeouts"],
            "reconnect": effective["reconnect"],
            "loss_after_registration": case.get("loss_after_registration", False),
            "settings": effective["recovery_policy"],
        }
        report["network"] = network
        save(directory / "network.json", network)
        _, reason = protect("before-workload", {})
        if reason:
            return report
        if case["id"] == "bandwidth-10kbit-per-vessel":
            verification = compose(
                project,
                file,
                "exec",
                "-T",
                "driver",
                "hub-network-agents",
                "--server",
                network["hub_address"],
                "--agents",
                "1",
                "--verify-bytes",
                "8192",
                timeout=60,
            )
            value = json.loads(verification.stdout.strip())
            save(directory / "bandwidth-verification.json", value)
            if value["seconds"] < 4:
                raise RuntimeError("8 KiB probe was not delayed by the per-vessel 10 kbit/s shaper")
            _, reason = protect("after-bandwidth-probe", {})
            if reason:
                return report
        log = (directory / "driver.jsonl").open("w")
        with (directory / "driver.stderr.log").open("w") as errors:
            args = [
                "docker",
                "compose",
                "-p",
                project,
                "-f",
                str(file),
                "exec",
                "-T",
                "driver",
                "hub-network-agents",
                "--server",
                network["hub_address"],
                "--agents",
                str(config["agents"]),
                "--hold-seconds",
                str(config["hold_seconds"]),
                "--setup-seconds",
                str(config["registration_budget_seconds"]),
                "--timing-probes",
                str(config["timing_probes"]),
            ]
            policy = effective["recovery_policy"]
            args += [
                "--reconnect",
                str(effective["reconnect"]).lower(),
                "--attempt-seconds",
                str(policy.get("attempt_seconds", 45)),
                "--backoff-base-ms",
                str(policy.get("backoff_base_ms", 1000)),
                "--backoff-max-ms",
                str(policy.get("backoff_max_ms", 30000)),
                "--retry-seed",
                str(policy.get("jitter_seed", 20261120)),
                "--source-ips",
                ",".join(RETRY_IPS if effective["reconnect"] else RETRY_IPS[:1]),
                "--admission-slots",
                str(policy.get("admission_slots", 100)),
                "--retry-slots",
                str(policy.get("reserved_retry_slots", 0)),
                "--borrow-admission",
                str(policy.get("borrow_admission", False)).lower(),
                "--response-timeout-seconds",
                str(policy.get("response_timeout_seconds", 0)),
                "--settle-seconds",
                str(config.get("settle_seconds", 0)),
                "--diagnostics",
                str(config.get("diagnostics", False)).lower(),
                "--followup-seconds",
                str(config.get("followup_seconds", 0) if case["id"] != "baseline" else 0),
            ]
            if case.get("loss_after_registration"):
                args += ["--hold-gate", "/results/impairment-applied.json"]
            process = subprocess.Popen(args, stdout=log, stderr=errors)
            watcher = ProtectionWatch(lambda: protect("monitor", {}))
            watcher.thread.start()
            previous_cpu = {}
            last_phase = None
            while process.poll() is None:
                rows = json_lines(directory / "driver.jsonl")
                latest = rows[-1] if rows else {}
                phase = latest.get("phase", "starting")
                if watcher.error:
                    raise watcher.error
                if report["outcome"].startswith("protection:"):
                    break
                sample, reason = protect(phase, latest)
                if reason:
                    break
                if phase == "awaiting-impairment" and not (directory / "impairment-applied.json").exists():
                    latest = observe_cohort(directory)
                    if report["outcome"].startswith("protection:") or watcher.error:
                        break
                    if latest.get("connected") != config["agents"]:
                        raise RuntimeError("Pre-registration cohort is incomplete; loss was not applied")
                    print("  Fleet registered; applying packet loss before the observation window", flush=True)
                    started_at = time.time()
                    invoke(
                        "exec",
                        "-T",
                        "delay",
                        "python3",
                        "/usr/local/lib/hub-network/tc_setup.py",
                        "change",
                        "--role",
                        "delay",
                        "--interfaces",
                        *network["interfaces"]["delay"],
                        "--agents",
                        str(config["agents"]),
                        "--timing-probes",
                        str(config["timing_probes"]),
                        "--delay-ms",
                        str(case["delay_ms_each_direction"]),
                        "--loss-percent",
                        str(case["loss_percent"]),
                        "--queue-packets",
                        str(config["shared_queue_packets"]),
                    )
                    sample, reason = protect(phase, latest)
                    if reason:
                        break
                    after_change = observe_cohort(directory)
                    if report["outcome"].startswith("protection:") or watcher.error:
                        break
                    if after_change.get("connected") != config["agents"]:
                        raise RuntimeError("Cohort became incomplete during shaping; observation gate withheld")
                    save(
                        directory / "impairment-applied.json",
                        {
                            "status": "applied",
                            "case": case,
                            "started_at": started_at,
                            "completed_at": time.time(),
                            "sample_before_change": latest,
                            "sample_after_change": after_change,
                        },
                    )
                stats = command(["docker", "stats", "--no-stream", "--format", "{{json .}}", *ids.values()], timeout=30)
                sample["containers"] = [json.loads(line) for line in stats.stdout.splitlines() if line.strip()]
                sample["accounting"] = {}
                for side in SERVICES:
                    accounting = json.loads(
                        invoke("exec", "-T", side, "python3", "/usr/local/lib/hub-network/monitor.py").stdout
                    )
                    if side in previous_cpu:
                        before = previous_cpu[side]
                        elapsed = accounting["monotonic_seconds"] - before["monotonic_seconds"]
                        delta = {
                            key: value - before["cpu_stat"].get(key, 0) for key, value in accounting["cpu_stat"].items()
                        }
                        accounting["interval"] = {
                            "seconds": elapsed,
                            "cpu_stat_delta": delta,
                            "user_cpu_percent": delta.get("user_usec", 0) / elapsed / 10000,
                            "system_cpu_percent": delta.get("system_usec", 0) / elapsed / 10000,
                        }
                    previous_cpu[side] = accounting
                    sample["accounting"][side] = accounting
                if phase != last_phase or len(report["resource_samples"]) % 5 == 0:
                    print(
                        f"  {case['id']}: {phase}; connected={latest.get('connected', 0)}/{config['agents']}; "
                        f"registered={latest.get('registered_total', 0)}; failures={latest.get('failures', 0)}",
                        flush=True,
                    )
                    last_phase = phase
                time.sleep(2)
        log.flush()
        watcher.close()
        observer_error = watcher.error
        watcher = None
        if observer_error:
            raise observer_error
        if not report["outcome"].startswith("protection:"):
            rows = json_lines(directory / "driver.jsonl")
            protect("finished", rows[-1] if rows else {})
        if not report["outcome"].startswith("protection:") and process.returncode != 0:
            raise RuntimeError(f"Driver exited {process.returncode}; inspect driver.stderr.log")
        measurements = [value for value in json_lines(directory / "driver.jsonl") if value.get("type") == "report"]
        if measurements and not report["outcome"].startswith("protection:"):
            report["measurement"] = measurements[-1]
            report["outcome"] = (
                "target-held"
                if measurements[-1]["target_held"]
                else "recorded-with-agent-loss-or-incomplete-registration"
            )
        elif not report["outcome"].startswith("protection:"):
            raise RuntimeError("Driver exited without a final report; inspect driver.stderr.log")
        for side in ("rate", "delay"):
            compose(
                project,
                file,
                "exec",
                "-T",
                side,
                "python3",
                "/usr/local/lib/hub-network/tc_setup.py",
                "collect",
                "--role",
                side,
                "--interfaces",
                *network["interfaces"][side],
                timeout=120,
            )
    except Exception as exc:  # noqa: BLE001 - keep failed setup and incomplete measurements reviewable
        report["harness_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if watcher:
            watcher.close()
        cleanup_errors = []
        try:
            logs = compose(project, file, "logs", "--no-color", "hub", timeout=30, check=False) if started else None
            if logs is not None and logs.returncode:
                raise RuntimeError(logs.stderr.strip() or "Hub log collection failed")
            if logs is not None:
                (directory / "hub.log").write_text(logs.stdout + logs.stderr)
        except Exception as exc:  # noqa: BLE001 - log failure must not prevent owned-resource cleanup
            report["log_collection_error"] = f"{type(exc).__name__}: {exc}"
            report.setdefault("harness_error", report["log_collection_error"])
        try:
            if started:
                verify_daemon(daemon)
            down = (
                compose(project, file, "down", "--volumes", "--remove-orphans", timeout=120, check=False)
                if started
                else None
            )
            if down is not None and down.returncode:
                report.setdefault("cleanup_diagnostics", []).append(down.stderr.strip() or "Compose teardown failed")
        except Exception as exc:  # noqa: BLE001 - retain teardown failure and still finalize the report
            report.setdefault("cleanup_diagnostics", []).append(f"Teardown: {type(exc).__name__}: {exc}")
        if process is not None:
            try:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=5)
            except Exception as exc:  # noqa: BLE001 - preserve cleanup evidence even if the driver cannot be reaped
                cleanup_errors.append(f"Driver: {type(exc).__name__}: {exc}")
        if log:
            log.close()
        try:
            if started:
                verify_daemon(daemon)
                owned_resources(project)
                verify_daemon(daemon)
        except Exception as exc:  # noqa: BLE001 - missing verification must never count as successful cleanup
            cleanup_errors.append(f"Verification: {type(exc).__name__}: {exc}")
        report["cleanup_errors"] = cleanup_errors
        report["cleanup_verified"] = not cleanup_errors
        report["elapsed_seconds"] = time.monotonic() - began
        save(directory / "report.json", report)
    return report


def configuration():
    return json.loads((ROOT / "settings.json").read_text())


def receipt_report(folder, output, signature):
    """Reuse only intact evidence belonging to this case's output directory."""
    try:
        receipt = json.loads((folder / "receipt.json").read_text())
        if not isinstance(receipt, dict) or receipt.get("signature") != signature:
            return None
        path = output / receipt["report"]
        if not path.resolve().is_relative_to(folder.resolve()) or digest(path) != receipt["sha256"]:
            return None
        report = json.loads(path.read_text())
        if (
            not isinstance(report, dict)
            or report.get("cleanup_verified") is not True
            or report.get("harness_error")
            or not isinstance(report.get("measurement"), dict)
            or type(report["measurement"].get("target_held")) is not bool
        ):
            return None
        return report
    except (OSError, ValueError, TypeError, KeyError):
        return None


def run(args):
    image = image_identity()
    config = configuration()
    selected = selected_cases(config, args.case)
    if args.agents is not None and args.command != "smoke":
        raise ValueError("--agents is only available for bounded smoke runs")
    if args.command == "smoke" and not 1 <= (4 if args.agents is None else args.agents) <= 9000:
        raise ValueError("Smoke fleet must be between 1 and 9000 agents")
    initial_host = workload_host(image)
    if args.command == "smoke":
        agents = 4 if args.agents is None else args.agents
        config.update(
            agents=agents,
            hold_seconds=120 if agents > 4 else 15,
            registration_budget_seconds=600 if agents > 4 else 120,
            minimum_available_memory_bytes=max(4 * 2**30, initial_host["total_memory_bytes"] // 4),
            timing_probes=2,
        )
        for case in config["cases"]:
            if "observation" in case:
                case["observation"].update(hold_seconds=config["hold_seconds"], settle_seconds=15 if agents > 4 else 2)
    else:
        if platform.system() != "Linux":
            raise RuntimeError("Full network evaluations require Linux; use smoke for a smaller local run")
        config["minimum_available_memory_bytes"] = max(
            config["minimum_available_memory_bytes"], initial_host["total_memory_bytes"] // 4
        )
    if initial_host["available_memory_bytes"] < config["minimum_available_memory_bytes"]:
        raise RuntimeError("Insufficient available device memory for the declared reserve")
    output = (
        args.output.resolve()
        if args.output
        else ROOT / "results" / (args.command + "-" + dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ"))
    )
    output.mkdir(parents=True, exist_ok=True)
    inputs = {
        "config": config,
        "image": image,
        "sources": {name: digest(ROOT / name) for name in SOURCES},
        "platform": platform.platform(),
        "workload": "Core-compatible Rust 5s WebSocket heartbeat; HTTP only",
        "bandwidth_scope": "10 kbit/s per vessel independently in each direction",
        "failover": False,
        "per_flow_source_ports": "20000+i; identified timing probes 45000+i; unimpaired control 50001",
        "network_version": "ipip-private-underlay",
    }
    if captured_build_identity(inputs["sources"]) != image.get("build_inputs"):
        raise RuntimeError("Captured helper build inputs differ from the pinned image; run prepare again")
    if (output / "inputs.json").exists() and json.loads((output / "inputs.json").read_text()) != inputs:
        raise RuntimeError("Different input identity; choose a fresh output directory")
    for previous in output.glob("*/attempt-*"):
        report_path = previous / "report.json"
        if not report_path.is_file() or json.loads(report_path.read_text()).get("cleanup_verified") is not True:
            raise RuntimeError(f"Unverified cleanup: {previous}; inspect and clean up before another run")
    save(output / "inputs.json", inputs)
    signature = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    for case in selected:
        for repetition in range(1, config["repetitions"] + 1):
            label = f"{case['id']}-r{repetition:02}"
            folder = output / label
            receipt = folder / "receipt.json"
            recorded = receipt_report(folder, output, signature)
            if recorded is not None:
                if case["id"] == "baseline" and not recorded["measurement"]["target_held"]:
                    raise RuntimeError("Recorded baseline did not hold the fleet; inspect before proceeding")
                print(f"SKIP verified: {label}", flush=True)
                continue
            attempt = 1
            while (folder / f"attempt-{attempt:02}").exists():
                attempt += 1
            directory = (folder / f"attempt-{attempt:02}").resolve()
            print(f"START {label}: {directory}", flush=True)
            report = execute(case, config, image, directory)
            print(f"DONE {label}: {report['outcome']} | cleanup={report['cleanup_verified']}", flush=True)
            if (
                not report["cleanup_verified"]
                or "harness_error" in report
                or report["outcome"].startswith("protection:")
            ):
                raise RuntimeError("Suite stopped on setup/instrument/device issue; completed cases remain preserved")
            if case["id"] == "baseline" and not report["measurement"]["target_held"]:
                raise RuntimeError("Baseline did not maintain the requested fleet; inspect before testing constraints")
            save(
                receipt,
                {
                    "signature": signature,
                    "report": str((directory / "report.json").relative_to(output)),
                    "sha256": digest(directory / "report.json"),
                },
            )


def package():
    output = ROOT / "packages/hub-network.tar.gz"
    output.parent.mkdir(exist_ok=True)
    with tarfile.open(output, "w:gz") as bundle:
        prefix = output.name.removesuffix(".tar.gz")
        for name in SOURCES:
            bundle.add(ROOT / name, arcname=f"{prefix}/{name}", recursive=False)
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "check", "smoke", "run", "package", "cleanup"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--agents", type=int, help="bounded smoke fleet size, at most 9000")
    args = parser.parse_args()
    (ROOT / ".runtime").mkdir(exist_ok=True)
    if args.command == "cleanup" and args.output is None:
        parser.error("cleanup requires --output")
    if args.command == "check" and args.output is None:
        args.output = ROOT / "results/check"
    elif args.command in ("run", "smoke") and args.output is None:
        args.output = ROOT / "results" / (args.command + "-" + dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    with output_lock(args.output if args.output else ROOT / ".runtime"):
        if args.command == "prepare":
            prepare()
        elif args.command == "package":
            package()
        elif args.command == "cleanup":
            cleanup_attempts(args.output.resolve())
        elif args.command == "check":
            for previous in args.output.glob("*/attempt-*"):
                path = previous / "report.json"
                if not path.is_file() or json.loads(path.read_text()).get("cleanup_verified") is not True:
                    raise RuntimeError("Unverified cleanup: " + str(previous))
            image = image_identity()
            code = (
                "import sys,subprocess;sys.path.insert(0,'/usr/local/lib/hub-network');import tc_setup;"
                "subprocess.run(['ip','tunnel','add','probe','mode','ipip','local','127.0.0.1',"
                "'remote','127.0.0.2'],check=True);"
                "commands='\\n'.join(tc_setup.rate_rules([20000],'driver',10)).replace('DEVICE','lo')+'\\n';"
                "subprocess.run(['tc','-batch','-'],input=commands,text=True,check=True);"
                "subprocess.run(['tc','qdisc','del','dev','lo','root'],check=True);"
                "commands='\\n'.join(tc_setup.delay_rules(250,30,65536)).replace('DEVICE','lo')+'\\n';"
                "subprocess.run(['tc','-batch','-'],input=commands,text=True,check=True);"
                "subprocess.run(['tc','-j','qdisc','show','dev','lo'],check=True)"
            )
            result = command(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--cap-add",
                    "NET_ADMIN",
                    "--network",
                    "none",
                    image["id"],
                    "python3",
                    "-c",
                    code,
                ]
            )
            print(result.stdout)
            config = json.loads((ROOT / "settings.json").read_text())
            initial_host = workload_host(image)
            config.update(
                agents=1,
                timing_probes=1,
                hold_seconds=1,
                registration_budget_seconds=10,
                minimum_available_memory_bytes=max(4 * 2**30, initial_host["total_memory_bytes"] // 4),
            )
            if initial_host["available_memory_bytes"] < config["minimum_available_memory_bytes"]:
                raise RuntimeError("Insufficient available device memory for the declared reserve")
            target = args.output.resolve() / "preflight"
            number = max((int(path.name.removeprefix("attempt-")) for path in target.glob("attempt-*")), default=0) + 1
            directory = target / f"attempt-{number:02}"
            report = execute(selected_cases(config, "baseline")[0], config, image, directory)
            print(f"Connectivity preflight: {report['outcome']}; cleanup={report['cleanup_verified']}; {directory}")
            if (
                not report["cleanup_verified"]
                or not report.get("measurement", {}).get("target_held")
                or report.get("harness_error")
            ):
                raise RuntimeError("Connectivity preflight failed; inspect its report and network diagnostics")
        else:
            run(args)


if __name__ == "__main__":
    main()
