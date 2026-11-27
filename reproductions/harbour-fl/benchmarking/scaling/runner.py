"""Manual, resumable 1/3/6/12/24-client scaling study against an existing application deployment."""

import contextlib
import datetime as dt
import fcntl
import json
import math
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from cli import parser
from cluster import Cluster, documents, preflight
from common import ROOT, digest, load, save, sha, tool_identity
from measurements import curves, overlaps_training, parse_rounds, summarise, write_csv
from partitions import prepare

UTC = dt.timezone.utc


def completion_time(content, observed):
    """Prefer the coordinator event clock; otherwise use the terminal log observation."""
    for line in reversed(content.splitlines()):
        start = line.find('{"')
        if start < 0:
            continue
        try:
            event = json.loads(line[start:])
            if event.get("kind") == "completed" and event.get("at"):
                value = event["at"]
                return (
                    dt.datetime.fromtimestamp(value, UTC)
                    if isinstance(value, (int, float))
                    else (dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC))
                )
        except (ValueError, TypeError, OSError):
            continue
    return observed


def validate(settings):
    """Reject unsupported designs before accessing Kubernetes."""
    for key in (
        "rounds",
        "repetitions",
        "local_epochs",
        "batch_size",
        "torch_threads",
        "sample_interval_seconds",
        "round_timeout_seconds",
        "job_timeout_seconds",
    ):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if any(type(count) is not int for count in settings["clients"]) or settings["clients"] not in (
        [1, 3, 6, 12],
        [1, 3, 6, 12, 24],
    ):
        raise ValueError("Supported participant counts are 1,3,6,12, optionally followed by 24")
    if settings["torch_threads"] != 1:
        raise ValueError("This CPU scaling design fixes one Torch thread per process")
    design = settings.get("study_design", "fixed_total")
    if design not in ("fixed_total", "fixed_per_client"):
        raise ValueError("Unknown scaling study design")
    if design == "fixed_per_client":
        for key in ("train_samples_per_client", "validation_samples_per_client"):
            if type(settings.get(key)) is not int or settings[key] < 1:
                raise ValueError(key + " must be a positive integer")


@contextlib.contextmanager
def locks(cluster):
    """Prevent overlapping scaling runs against the same Harbour-FL application."""
    with contextlib.ExitStack() as stack:
        for path in (ROOT / "active.lock", cluster.app / "benchmarking/active.lock"):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.open("a+"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another benchmark holds " + str(path) + "; wait for it to finish") from exc
        yield


def jobs(settings, smoke=False):
    """Order smaller independent points first; keep the smoke separate from measured curve points."""
    if smoke:
        return [("smoke-c01-r01", 1, 1)]
    return [
        (f"c{count:02d}-r{rep:02d}", count, rep)
        for count in settings["clients"]
        for rep in range(1, settings["repetitions"] + 1)
    ]


def verified(folder, signature):
    """Trust a completion only when configuration, cleanup and every saved artifact still match."""
    marker = folder / "completion.json"
    if not marker.is_file():
        return False
    try:
        value = load(marker)
        if not isinstance(value, dict):
            return False
        artifacts = value.get("artifacts", {})
        return (
            value.get("signature") == signature
            and value.get("cleanup_verified") is True
            and isinstance(artifacts, dict)
            and bool(artifacts)
            and all(
                (folder / name).resolve().is_relative_to(folder.resolve())
                and (folder / name).is_file()
                and sha(folder / name) == checksum
                for name, checksum in artifacts.items()
            )
        )
    except (OSError, ValueError, TypeError):
        return False


def progress(batch, completed, reports):
    """Show an explicitly provisional ETA between jobs, updated from recorded point runtimes."""
    remaining = [j for j in batch if j[0] not in completed]
    timings = [r["elapsed_seconds"] for key, r in reports.items() if not key.startswith("smoke-")]
    per_job = sum(timings) / len(timings) if timings else 1800
    seconds = per_job * len(remaining)
    finish = dt.datetime.now().astimezone() + dt.timedelta(seconds=seconds)
    print(
        f"Batch: {len(batch) - len(remaining)}/{len(batch)} recorded | Remaining: {len(remaining)} | "
        f"ETA: ~{math.ceil(seconds / 60)}m | Estimated finish: {finish:%Y-%m-%d %H:%M %Z}"
        + (" (planning estimate)" if not timings and remaining else ""),
        flush=True,
    )


def wait_clients(cluster, names, timeout=300):
    """Wait for actual dataset loading, beyond a container merely becoming Ready."""
    deadline = time.monotonic() + timeout
    pending = set(names)
    while pending:
        for name in list(pending):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Dataset initialization timed out: " + ", ".join(sorted(pending)))
            content = cluster.logs(name, timeout=min(30, remaining))
            if time.monotonic() >= deadline:
                raise TimeoutError("Dataset initialization timed out: " + ", ".join(sorted(pending)))
            if "worker initialization failed" in content or "Traceback (most recent call last)" in content:
                raise RuntimeError("Client initialization failed: " + name)
            if "validation pairs" in content:
                pending.remove(name)
        if pending:
            if time.monotonic() >= deadline:
                raise TimeoutError("Dataset initialization timed out: " + ", ".join(sorted(pending)))
            print("Waiting for dataset loading: " + ", ".join(sorted(pending)), flush=True)
            time.sleep(min(5, max(0, deadline - time.monotonic())))


def wait_rollouts(cluster, names, timeout=300):
    """Print startup progress while Kubernetes schedules pods and binds the independent storage claim."""
    deadline = time.monotonic() + timeout
    pending = set(names)
    while pending:
        for name in list(pending):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Rollout timed out; see saved pods and events")
            value = cluster.get("deployment/" + name, timeout=min(60, remaining))
            if time.monotonic() >= deadline:
                raise TimeoutError("Rollout timed out; see saved pods and events")
            status = value.get("status", {})
            if (
                status.get("observedGeneration", 0) >= value["metadata"]["generation"]
                and status.get("availableReplicas", 0) == 1
            ):
                pending.remove(name)
        if pending:
            print("Waiting for rollout: " + ", ".join(sorted(pending)), flush=True)
            if time.monotonic() >= deadline:
                raise TimeoutError("Rollout timed out; see saved pods and events")
            time.sleep(min(5, max(0, deadline - time.monotonic())))


def image_check(cluster, snapshot, count, bound=None):
    """Reject stale or retagged local images instead of silently mixing Core/application builds."""
    pods = cluster.get("pods")["items"]
    expected = {
        "ws-server": snapshot["images"]["ws-server"]["image_id"],
        "vessel-coordinator": snapshot["images"]["vessel-coordinator"]["image_id"],
        **{f"vessel-{i + 1}": snapshot["images"]["vessel-1"]["image_id"] for i in range(count)},
    }
    seen = {}
    for pod in pods:
        role = pod["metadata"].get("labels", {}).get("app")
        if role in expected:
            statuses = pod.get("status", {}).get("containerStatuses", [])
            if len(statuses) != 1 or statuses[0]["imageID"] != expected[role]:
                raise RuntimeError("Running image differs from the source image: " + role)
            if statuses[0].get("restartCount", 0) or pod["metadata"].get("deletionTimestamp"):
                raise RuntimeError("Pod restarted or is terminating; clean scaling point invalidated: " + role)
            identity = {"uid": pod["metadata"]["uid"], "name": pod["metadata"]["name"]}
            if role in seen or (bound is not None and identity != bound.get(role)):
                raise RuntimeError("Scaling pod replaced or duplicated: " + role)
            seen[role] = identity
    if set(seen) != set(expected):
        raise RuntimeError("Not every scaling role has a running image")
    return seen


def execute(cluster, settings, snapshot, folder, key, count, repetition, capacity=None, *, target=None):
    """Always retain logs and cleanup evidence; a failed point cannot become a resumable success."""
    target = cluster.target() if target is None else target
    folder.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    attempt_id = f"scaling-{key}-{dt.datetime.now(UTC):%Y%m%d%H%M%S%f}"
    data, partition = prepare(snapshot, count, settings["seed"], study=settings)
    manifests = documents(settings, snapshot, data, count, attempt_id)
    report = {
        "study_design": settings.get("study_design", "fixed_total"),
        "job_id": attempt_id,
        "clients": count,
        "repetition": repetition,
        "status": "failed",
        "cleanup_verified": False,
        "started_at": dt.datetime.now(UTC).isoformat(),
    }
    logs, samples, sample_errors = {}, [], []
    names = ["ws-server", *(f"vessel-{i + 1}" for i in range(count)), "vessel-coordinator"]
    adapter = getattr(cluster, "adapter", None)
    if adapter is not None:
        adapter.config = replace(adapter.config, workers=tuple(names[1:-1]))
        report["effective_scenario"] = adapter.prepare_scenario(
            {
                **{key: settings[key] for key in ("rounds", "local_epochs", "batch_size", "torch_threads")},
                "training_seed": settings["seed"],
                "round_timeout_seconds": settings["round_timeout_seconds"],
                "minimum_clients": count,
                "device": "cpu",
                "recovery_mode": "disabled",
                "fault": {"type": "none"},
            }
        )
    with tempfile.TemporaryDirectory(dir=folder.parent, prefix=".attempt-") as staging:
        journal = Path(staging) / "published"
        save(journal / "run-target.json", target)
        if capacity is not None:
            save(journal / "capacity.json", capacity)
        save(journal / "partitions.json", partition)
        save(journal / "manifests.json", manifests)
        save(journal / "report.json", {**report, "phase": "pre-start", "cleanup_verified": True})
        journal.rename(folder)
    original_error = None
    bound = None

    def check_pods():
        if bound:
            image_check(cluster, snapshot, count, bound)

    def role_logs(name):
        return cluster.logs(name, pod=bound[name]["name"]) if bound else cluster.logs(name)

    try:
        save(folder / "report.json", report)
        cluster.cleanup()
        cluster.namespace_journal = folder / "namespace-owner.json"
        print("Creating independent hub and CPU client deployments", flush=True)
        cluster.apply(manifests)
        wait_rollouts(cluster, names[:-1])
        wait_clients(cluster, names[1:-1])
        cluster.call("scale", "deployment/vessel-coordinator", "--replicas=1")
        wait_rollouts(cluster, ["vessel-coordinator"])
        bound = image_check(cluster, snapshot, count)
        report["pods"] = bound
        seen = 0
        training_started = time.monotonic()
        training_wall_start = dt.datetime.now(UTC)
        deadline = training_started + settings["job_timeout_seconds"]
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Scaling point exceeded its overall timeout")
            check_pods()
            content = role_logs("vessel-coordinator")
            terminal_observed = dt.datetime.now(UTC)
            logs["vessel-coordinator"] = content
            rounds, _ = parse_rounds({"vessel-coordinator": content}, adapter, settings["rounds"])
            print(
                f"[{int(time.monotonic() - started) // 60:02d}:{int(time.monotonic() - started) % 60:02d}] "
                f"{key}: {len(rounds)}/{settings['rounds']} rounds complete",
                flush=True,
            )
            if len(rounds) > seen:
                seen = len(rounds)
                for i in sorted(rounds):
                    print(
                        f"  round {i}: {rounds[i]['round_work_seconds']:.1f}s; "
                        f"clients={rounds[i]['participating_clients']:g}",
                        flush=True,
                    )
            try:
                sample = cluster.resources()
                pod_names = {pod["name"] for pod in bound.values()}
                sample["items"] = [p for p in sample.get("items", []) if p["metadata"]["name"] in pod_names]
                sample["observed_elapsed_seconds"] = time.monotonic() - training_started
                samples.append(sample)
            except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as exc:
                sample_errors.append(str(exc))
            check_pods()
            if time.monotonic() >= deadline:
                raise TimeoutError("Scaling point exceeded its overall timeout")
            if "harbour FL coordinator failed" in content or "worker initialization failed" in content:
                raise RuntimeError("Coordinator failed; see its saved log")
            if '"kind": "completed"' in content:
                if len(rounds) != settings["rounds"]:
                    raise RuntimeError("Coordinator completed with missing rounds")
                training_wall_end = completion_time(content, terminal_observed)
                break
            time.sleep(min(settings["sample_interval_seconds"], max(0, deadline - time.monotonic())))
        pod_names = {pod["name"] for pod in bound.values()}

        def covered():
            return pod_names.issubset(
                {
                    item["metadata"]["name"]
                    for sample in samples
                    for item in sample.get("items", [])
                    if item.get("containers")
                }
            )

        grace = min(deadline, time.monotonic() + 30)
        while not covered() and time.monotonic() < grace:
            time.sleep(min(1, max(0, (grace - time.monotonic()) / 2)))
            if time.monotonic() >= grace:
                break
            check_pods()
            remaining = grace - time.monotonic()
            if remaining <= 0:
                break
            try:
                sample = cluster.resources(timeout=min(20, remaining))
                if time.monotonic() >= grace:
                    break
                sample["items"] = [
                    item
                    for item in sample.get("items", [])
                    if item["metadata"]["name"] in pod_names
                    and overlaps_training(item, training_wall_start, training_wall_end)
                ]
                sample["observed_elapsed_seconds"] = time.monotonic() - training_started
                sample["phase"] = "post-completion"
                samples.append(sample)
            except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as exc:
                sample_errors.append(str(exc))
        if time.monotonic() >= deadline:
            raise TimeoutError("Scaling point exceeded its overall timeout")
        check_pods()
        for name in names:
            logs[name] = role_logs(name)
        check_pods()
        report["summary"] = summarise(logs, samples, partition, settings, adapter)
        write_csv(folder / "rounds.csv", report["summary"]["rounds"])
        report["status"] = "passed"
    except BaseException as exc:  # noqa: BLE001 - retain the original failure or interruption before cleanup.
        original_error = exc
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            for name, content in logs.items():
                (folder / (name + ".log")).write_text(content)
            if report["status"] != "passed":
                for name in names:
                    try:
                        (folder / (name + ".diagnostic.log")).write_text(cluster.logs(name))
                    except Exception as exc:  # noqa: BLE001 - one unavailable log must not prevent cleanup.
                        report.setdefault("collection_errors", []).append(name + ": " + str(exc))
            check_pods()
            save(folder / "resource-samples.json", {"samples": samples, "errors": sample_errors})
            save(folder / "pods-final.json", cluster.get("pods"))
            save(folder / "events.json", cluster.get("events"))
            check_pods()
            # Retain a provisional report before cleanup, then record the verified deletion below.
            report["elapsed_seconds"] = time.monotonic() - started
            save(folder / "report.json", report)
        except BaseException as exc:  # noqa: BLE001 - interruption must not prevent namespace cleanup.
            report.setdefault("collection_errors", []).append(str(exc))
            report["status"] = "failed"
            if original_error is None:
                original_error = exc
        try:
            cluster.cleanup()
            report["cleanup_verified"] = True
        except Exception as exc:  # noqa: BLE001 - preserve the main error when cleanup also fails.
            report["cleanup_error"] = str(exc)
            report["status"] = "failed"
            if original_error is None:
                original_error = exc
        report["finished_at"] = dt.datetime.now(UTC).isoformat()
        report["elapsed_seconds"] = time.monotonic() - started
        try:
            save(folder / "report.json", report)
        except Exception as exc:  # noqa: BLE001 - preserve the main error if final report writing fails.
            if original_error is None:
                original_error = exc
    if original_error:
        raise original_error
    return report


def main():
    """Execute the selected scaling command and retain per-run evidence."""
    args = parser().parse_args()
    if args.action == "smoke" and args.case not in (None, 1):
        raise ValueError("Smoke supports only --case 1")
    settings = load(args.settings)
    validate(settings)
    if (
        args.action in ("smoke", "run")
        and args.case is not None
        and not any(count == args.case for _, count, _ in jobs(settings, args.action == "smoke"))
    ):
        raise ValueError("Selected case matches no jobs")
    design = settings.get("study_design", "fixed_total")
    default_output = ROOT / "results" / ("per-client" if design == "fixed_per_client" else "current")
    output = (args.output or default_output).expanduser().resolve()
    if args.action == "plan":
        print(
            json.dumps(
                {
                    "jobs": jobs(settings),
                    "settings": settings,
                    "scope": "Single-host, one hub, " + design + ", CPU only",
                    "initial_planning_estimate": "2-4 hours including preparation and smoke; hardware dependent",
                },
                indent=2,
            )
        )
        return
    if args.action == "status":
        for folder in sorted(output.glob("*/attempt-*")):
            path = folder / "report.json"
            if not path.is_file():
                print(folder.parent.name, folder.name, "interrupted; cleanup unverified")
                continue
            value = load(path)
            print(
                path.parent.parent.name,
                path.parent.name,
                value["status"],
                "cleanup=" + str(value["cleanup_verified"]),
                value.get("error", ""),
            )
        return
    cluster = Cluster(settings)
    with locks(cluster):
        if args.action == "cleanup":
            target = cluster.target()
            pending = []
            for folder in output.glob("*/attempt-*"):
                path = folder / "report.json"
                report = load(path) if path.is_file() else {"status": "interrupted"}
                if report.get("cleanup_verified"):
                    continue
                identity = folder / "run-target.json"
                if not identity.is_file() or load(identity) != target:
                    raise RuntimeError("Cleanup target differs or is unknown: " + str(folder))
                owner_path = folder / "namespace-owner.json"
                owner = load(owner_path) if owner_path.is_file() else None
                pending.append((path, report, owner))
            for _path, _report, owner in pending or [(None, None, None)]:
                cluster.namespace_owner = owner
                cluster.cleanup()
            for path, report, _owner in pending:
                report.update(cleanup_verified=True, cleanup_actor="explicit scaling cleanup command")
                save(path, report)
            print("Scaling namespace deletion verified; original deployments untouched")
            return
        for folder in output.glob("*/attempt-*"):
            path = folder / "report.json"
            if not path.is_file() or not load(path).get("cleanup_verified"):
                raise RuntimeError("Unverified cleanup: " + str(folder) + "; inspect and use mise run scaling cleanup")
        target = cluster.target()
        snapshot = preflight(cluster)
        for count in settings["clients"]:
            data, _partition = prepare(snapshot, count, settings["seed"], study=settings)
            documents(settings, snapshot, data, count, "preflight")
        output.mkdir(parents=True, exist_ok=True)
        tool = tool_identity()
        signature = digest({"tool": tool, "settings": settings, "source": snapshot["identity"], "target": target})
        inputs = {"signature": signature, "settings": settings, "snapshot": snapshot, "tool": tool, "target": target}
        previous = output / "inputs.json"
        if previous.is_file() and load(previous)["signature"] != signature:
            raise RuntimeError("Inputs changed; use a new --output directory rather than combine different studies")
        save(previous, inputs)
        if args.action == "check":
            print("PASS: images, dataset assets/splits, all partition sizes, Metrics API and manifest generation")
            print("No source deployments changed. Evidence: " + str(previous))
            return
        smoke_key = "smoke-c01-r01"
        smoke_ok = any(verified(p, signature) for p in (output / smoke_key).glob("attempt-*"))
        if args.action == "run" and not smoke_ok:
            raise RuntimeError("Run mise run scaling smoke first using the same settings and output")
        batch = jobs(settings, args.action == "smoke")
        if args.case:
            batch = [j for j in batch if j[1] == args.case]
        reports = {}
        for key, _, _ in jobs(settings) + jobs(settings, True):
            for folder in sorted((output / key).glob("attempt-*")):
                if verified(folder, signature):
                    reports[key] = load(folder / "report.json")
        progress(batch, set(reports), reports)
        for key, count, rep in batch:
            if key in reports:
                print("SKIP verified: " + key, flush=True)
                continue
            attempts = list((output / key).glob("attempt-*"))
            folder = output / key / f"attempt-{len(attempts) + 1:02d}"
            from capacity import check_capacity

            capacity = check_capacity(cluster, settings, snapshot, count, reports)
            print("START " + key + " | evidence: " + str(folder), flush=True)
            selected = {**settings, "rounds": 1} if key.startswith("smoke-") else settings
            report = execute(cluster, selected, snapshot, folder, key, count, rep, capacity=capacity, target=target)
            save(
                folder / "completion.json",
                {
                    "signature": signature,
                    "cleanup_verified": True,
                    "artifacts": {
                        p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name != "completion.json"
                    },
                },
            )
            reports[key] = report
            curves(output, reports)
            print(f"DONE {key} | {report['elapsed_seconds'] / 60:.1f}m | cleanup verified", flush=True)
            progress(batch, set(reports), reports)
        curves(output, reports)


if __name__ == "__main__":

    def terminate(_signum, _frame):
        """Allow the experiment's cleanup block to run on a normal termination request."""
        raise KeyboardInterrupt("Termination requested")

    signal.signal(signal.SIGTERM, terminate)
    try:
        main()
    except Exception as error:  # noqa: BLE001 - CLI boundary prints the retained experiment error.
        print(f"Scaling study stopped: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
