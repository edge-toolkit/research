"""Prepare Core runtime images and run resumable hub-failover evaluations."""

import argparse
import asyncio
import csv
import fcntl
import hashlib
import json
import os
import signal
import tarfile
import tempfile
import time
import urllib.error
from pathlib import Path

from topology import Deployment, command, docker_prefix, preflight, resolve_docker, save, verify_absent
from workload import exercise, http

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = (
    "study.py",
    "topology.py",
    "workload.py",
    "settings.json",
    "fleet-25000.json",
    "catalog.json",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "mise.toml",
    "mise.lock",
    ".gitignore",
    "tests/test_study.py",
    "tests/test_lifecycle.py",
    "haproxy.cfg",
)


def digest(path):
    """Hash evidence incrementally without loading large files into memory."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def prepare():
    """Record the shared Core release image without building or exporting Core source."""
    runtime = ROOT / ".runtime"
    docker = docker_prefix(resolve_docker())
    image = "et-ws-server-core:local"
    row = json.loads(command([*docker, "image", "inspect", image]))[0]
    manifest = {
        "image": image,
        "image_id": row["Id"],
        "architecture": row["Architecture"],
        "revision": (row["Config"].get("Labels") or {}).get("org.opencontainers.image.revision"),
    }
    save(runtime / "snapshot.json", manifest)
    settings = json.loads((ROOT / "settings.json").read_text())
    for key, image in settings.items():
        if key.endswith("_image"):
            command([*docker, "pull", image], timeout=180)
    print("Prepared " + manifest["image_id"], flush=True)


def signature(settings, snapshot):
    """Bind skipped results to exact study code, effective parameters and immutable runtime images."""
    files = {name: digest(ROOT / name) for name in SOURCE_FILES if (ROOT / name).is_file()}
    return hashlib.sha256(json.dumps([files, settings, snapshot], sort_keys=True).encode()).hexdigest()


async def attempt(settings, snapshot, output, *, operation=exercise, expected_images=None, docker_target=None):
    """Always finalize an evidence report and try cleanup, preserving the primary failure diagnostic."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".attempt-") as staging:
        published = Path(staging) / "published"
        save(published / "startup.json", {"phase": "pre-start"})
        published.rename(output)
    settings = dict(settings)
    report = {
        "schema_version": 1,
        "started_at": time.time(),
        "settings": settings,
        "snapshot": snapshot,
        "status": "not-established",
        "cleanup_verified": False,
        "cleanup_errors": [],
    }
    project = None
    pending = None

    async def compose(*args):
        nonlocal pending
        pending = asyncio.create_task(asyncio.to_thread(project.compose, *args))
        try:
            return await asyncio.shield(pending)
        finally:
            if pending.done():
                pending = None

    try:
        report["preflight"] = await asyncio.to_thread(preflight, settings, snapshot, docker_target=docker_target)
        if expected_images is not None and image_ids(report) != expected_images:
            raise RuntimeError("Attempt images differ from the suite's immutable image identity")
        if "source_ports" in report["preflight"]:
            settings["source_ports"] = report["preflight"]["source_ports"]
        report["docker_target"] = report["preflight"]["docker_target"]
        project = Deployment(settings, snapshot["image_id"], output, docker_target=report["docker_target"])
        report["project"] = project.name
        save(output / "report.json", report)
        save(
            output / "startup.json",
            {"phase": "starting", "project": project.name, "docker_target": report["docker_target"]},
        )
        print("Starting isolated hub A, standby B, NATS, storage and gateway", flush=True)
        await compose("up", "-d", "nats", "storage")
        deadline = time.monotonic() + 120
        while True:
            try:
                await asyncio.to_thread(http, f"http://127.0.0.1:{settings['port_base'] + 4}/healthz")
                try:
                    await asyncio.to_thread(http, f"http://127.0.0.1:{settings['port_base'] + 5}/")
                except urllib.error.HTTPError as exc:
                    if exc.code != 403:  # An unauthenticated S3 listing is refused once storage is serving.
                        raise
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError("Broker/storage readiness exceeded two minutes") from None
                await asyncio.sleep(1)
        await compose("up", "-d", "hub-a", "hub-b")
        deadline = time.monotonic() + 120
        for offset in (1, 2, 0):
            if offset == 0:
                await compose("up", "-d", "gateway")
            while True:
                try:
                    await asyncio.to_thread(http, f"http://127.0.0.1:{settings['port_base'] + offset}/health")
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise RuntimeError("Hub/gateway readiness exceeded two minutes") from None
                    await asyncio.sleep(1)
        await operation(settings, project, report)
        report["status"] = "passed"
    except asyncio.CancelledError:
        report["error"] = "Cancelled; owned deployment cleanup attempted"
        raise
    except Exception as exc:  # noqa: BLE001 - preserve any workload failure in the finalized attempt report
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:

        async def finalize():
            for task in (pending, getattr(project, "pending_pause", None) if project else None):
                if not isinstance(task, asyncio.Task):
                    continue
                try:
                    await asyncio.shield(task)
                except Exception as exc:  # noqa: BLE001 - retain pending command failures separately from removal
                    report.setdefault("command_errors", []).append("In-flight Compose: " + str(exc))
            if project:
                try:
                    report["cleanup_errors"].extend(await asyncio.to_thread(project.cleanup))
                    report["cleanup_verified"] = not report["cleanup_errors"]
                except Exception as exc:  # noqa: BLE001 - cleanup errors must not replace the workload failure
                    report["cleanup_errors"].append(str(exc))
                report["collection_errors"] = getattr(project, "collection_errors", [])
                report["cleanup_diagnostics"] = getattr(project, "cleanup_diagnostics", [])
            else:
                report["cleanup_verified"] = True
            if not report["cleanup_verified"]:
                report["status"] = "not-established"
            report["finished_at"] = time.time()
            save(output / "report.json", report)

        shutdown = asyncio.create_task(finalize())
        try:
            await asyncio.shield(shutdown)
        except asyncio.CancelledError:
            report["status"] = "not-established"
            report["error"] = "Cancelled; owned deployment cleanup attempted"
            await asyncio.shield(shutdown)
            raise
    return report


def reconcile(report_path):
    """Reverify a manually cleaned project on its recorded daemon without changing successful evidence."""
    marker = report_path.parent / "startup.json"
    if marker.is_file() and json.loads(marker.read_text()).get("phase") == "pre-start":
        return True
    if not report_path.is_file():
        return False
    report = json.loads(report_path.read_text())
    if report.get("cleanup_verified") is True:
        return True
    if not report.get("docker_target"):
        return False
    try:
        errors = verify_absent(report.get("project", ""), report["docker_target"])
    except Exception as exc:  # noqa: BLE001 - a failed ownership check must continue blocking resume
        errors = [str(exc)]
    report["cleanup_reconciliation"] = {"checked_at": time.time(), "errors": errors}
    report["cleanup_verified"] = not errors
    if not errors:
        (report_path.parent / "private/compose.json").unlink(missing_ok=True)
    save(report_path, report)
    return not errors


def image_ids(report):
    """Compare image content identities independently of tag spelling."""
    return {item["id"] for item in report["preflight"]["images"].values()}


def verified(output, identity):
    """Skip only successful cases whose full report hash and experiment identity remain unchanged."""
    receipt = output / "receipt.json"
    if not receipt.exists():
        return False
    try:
        row = json.loads(receipt.read_text())
        if row["signature"] != identity:
            return False
        report = output / row["report"]
        return (
            report.resolve().is_relative_to(output.resolve()) and report.is_file() and row["sha256"] == digest(report)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


async def suite(settings, snapshot, output, *, smoke=False):
    """Hold the output lock through execution, cleanup reconciliation and receipt publication."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return await run_suite(settings, snapshot, output, smoke=smoke)


async def run_suite(settings, snapshot, output, *, smoke=False):
    """Run the short smoke first, then independent repetitions; retain completed evidence across retries."""
    output.mkdir(parents=True, exist_ok=True)
    admitted = {**settings, "fleet_size": 10, "warmup_seconds": 10} if smoke else settings
    check = preflight(admitted, snapshot)
    expected_images = {item["id"] for item in check["images"].values()}
    pinned = {
        **settings,
        **{key: check["images"][value]["id"] for key, value in settings.items() if key.endswith("_image")},
    }
    identity = signature(
        settings, {**snapshot, "runtime_images": check["images"], "docker_target": check["docker_target"]}
    )
    inputs = output / "inputs.json"
    if inputs.is_file() and json.loads(inputs.read_text()).get("signature") != identity:
        raise RuntimeError("Inputs changed; choose a new output directory")
    for previous in output.glob("*/attempt-*"):
        report_path = previous / "report.json"
        if not reconcile(report_path):
            raise RuntimeError(f"Unverified cleanup: {previous}; inspect and clean up before another run")
    save(
        output / "inputs.json", {"signature": identity, "settings": settings, "snapshot": snapshot, "preflight": check}
    )
    cases = [("smoke", {**pinned, "fleet_size": 10, "warmup_seconds": 10})]
    if not smoke:
        cases += [(f"r{i:02}", pinned) for i in range(1, settings["repetitions"] + 1)]
    timings = []
    for index, (name, effective) in enumerate(cases):
        target = output / name
        if verified(target, identity):
            print("SKIP verified: " + name, flush=True)
            continue
        target.mkdir(exist_ok=True)
        numbers = [int(p.name.split("-")[1]) for p in target.glob("attempt-*")]
        attempt_dir = target / f"attempt-{max(numbers, default=0) + 1:02}"
        estimate = sum(timings) / len(timings) if timings else settings["warmup_seconds"] + 120
        print(
            f"START {name} ({index + 1}/{len(cases)}); ETA ~{estimate * (len(cases) - index) / 60:.1f}m; "
            f"evidence: {attempt_dir}",
            flush=True,
        )
        began = time.monotonic()
        report = await attempt(
            effective, snapshot, attempt_dir, expected_images=expected_images, docker_target=check["docker_target"]
        )
        timings.append(time.monotonic() - began)
        if report["status"] != "passed":
            raise RuntimeError(
                f"{name} not established: {report.get('error', report['cleanup_errors'])}; see {attempt_dir}"
            )
        if image_ids(report) != expected_images:
            raise RuntimeError("Attempt image evidence differs from the suite signature; receipt not written")
        path = attempt_dir / "report.json"
        save(
            target / "receipt.json",
            {"signature": identity, "report": str(path.relative_to(target)), "sha256": digest(path)},
        )
        export_csv(output)
        print(f"DONE {name}: recovery {report['timing']['recovery_seconds']:.3f}s; cleanup verified", flush=True)
    export_csv(output)


def export_csv(output):
    """Export one row per verified repetition while keeping raw monotonic timing and load samples in JSON."""
    with (output / "summary.csv").open("w", newline="") as stream:
        identity = json.loads((output / "inputs.json").read_text())["signature"]
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "case",
                "fleet_size",
                "recovery_seconds",
                "detection_seconds",
                "post_down_seconds",
                "fleet_recovery_seconds",
                "core_image_id",
            ],
        )
        writer.writeheader()
        for receipt in sorted(output.glob("*/receipt.json")):
            if not verified(receipt.parent, identity):
                continue
            row = json.loads(receipt.read_text())
            path = receipt.parent / row["report"]
            report = json.loads(path.read_text())
            writer.writerow(
                {
                    "case": receipt.parent.name,
                    "fleet_size": report["settings"]["fleet_size"],
                    **{
                        key: report["timing"][key]
                        for key in ("recovery_seconds", "detection_seconds", "post_down_seconds")
                    },
                    "fleet_recovery_seconds": report["fleet_recovery_seconds"],
                    "core_image_id": report["snapshot"]["image_id"],
                }
            )


def package():
    """Bundle study inputs; the shared Core release is installed separately."""
    output = ROOT / "packages/hub-failover.tar.gz"
    output.parent.mkdir(exist_ok=True)
    prefix = output.name.removesuffix(".tar.gz")
    with tarfile.open(output, "w:gz") as bundle:
        for name in SOURCE_FILES:
            bundle.add(ROOT / name, arcname=prefix + "/" + name)
    print(output)


def large_fleet(settings):
    """Use the same checked source-port policy for CLI and controller fleets."""
    return {
        **settings,
        **json.loads((ROOT / "fleet-25000.json").read_text()),
        **{key: value for key, value in settings.items() if key.startswith("source_port_")},
    }


def checked_ports(settings):
    """Admit large explicit fleet requests with the same checked pool as the named preset."""
    result = dict(settings)
    if result["fleet_size"] >= 16384:
        preset = json.loads((ROOT / "fleet-25000.json").read_text())
        for key in ("source_port_base", "source_port_limit"):
            result.setdefault(key, preset[key])
    return result


def adapter_result(output):
    """Expose JSON recovery incidents to the controller, with CSV kept as a separate artifact."""
    identity = json.loads((output / "inputs.json").read_text())["signature"]
    reports = []
    for receipt in sorted(output.glob("*/receipt.json")):
        if receipt.parent.name != "smoke" and verified(receipt.parent, identity):
            entry = json.loads(receipt.read_text())
            reports.append(json.loads((receipt.parent / entry["report"]).read_text()))
    incidents = []
    for report in reports:
        if report.get("status") == "passed":
            timing = report["timing"]
            incidents.append(
                {
                    "workload": "hub-a",
                    "status": "recovered",
                    "observed_failure_at": timing["unhealthy_observed_at"],
                    "detected_at": timing["down_observed_at"],
                    "recovered_at": timing["work_restored_at"],
                }
            )
    evidence = {
        "schema_version": 1,
        "status": "passed",
        "recovery": {"provider": "haproxy", "clock_basis": "observer-monotonic-seconds", "incidents": incidents},
        "artifacts": {"summary_csv": str(output / "summary.csv")},
    }
    path = output / "controller-report.json"
    save(path, evidence)
    return {**evidence, "report": str(path)}


def main():
    """Expose preparation, preflight, short smoke and resumable repetitions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "package",
            "check",
            "smoke",
            "run",
            "adapter",
        ),
    )
    parser.add_argument("--fleet-size", type=int)
    parser.add_argument("--scenario", choices=("fleet-25000",))
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / "results/current"
    args.output = args.output.resolve()
    if args.action == "prepare":
        prepare()
        return
    if args.action == "package":
        package()
        return
    settings = json.loads((ROOT / "settings.json").read_text())
    if args.scenario:
        if args.action not in ("check", "smoke", "run"):
            parser.error("scenario is available for check, smoke and run")
        settings = large_fleet(settings)
    if args.fleet_size is not None:
        settings["fleet_size"] = args.fleet_size
    if args.repetitions is not None:
        settings["repetitions"] = args.repetitions
    settings = checked_ports(settings)
    if args.action == "adapter":
        settings = large_fleet(settings)
        request = json.loads(Path(os.environ["BENCHMARK_REQUEST_FILE"]).read_text())
        settings["fleet_size"] = request["parameters"]["fleet_size"]
        settings["repetitions"] = 1  # The shared controller owns repetition scheduling.
        args.output = Path(os.environ["BENCHMARK_OUTPUT_DIR"]) / f"repetition-{request['repetition']:02}"
    if not 1 <= settings["fleet_size"] <= 25000 or not 1 <= settings["repetitions"] <= 20:
        parser.error("fleet size must be 1..25000 and repetitions 1..20")
    snapshot = json.loads((ROOT / ".runtime/snapshot.json").read_text())
    if args.action == "check":
        checked = preflight(settings, snapshot)
        if "source_ports" in checked:
            ports = checked.pop("source_ports")
            checked["source_port_allocation"] = {"count": len(ports), "first": min(ports), "last": max(ports)}
        print(json.dumps(checked, indent=2))
    else:
        asyncio.run(run_interruptible(settings, snapshot, args.output, smoke=args.action == "smoke"))
        if args.action == "adapter":
            save(
                Path(os.environ["BENCHMARK_RESULT_FILE"]),
                adapter_result(args.output),
            )


async def run_interruptible(settings, snapshot, output, *, smoke=False):
    """Cancel once on SIGINT/SIGTERM and keep attempt finalization alive until it has completed."""
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(suite(settings, snapshot, output, smoke=smoke))
    stopping = False

    def request_stop():
        nonlocal stopping
        if not stopping:
            stopping = True
            task.cancel()

    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, request_stop)
    try:
        return await task
    finally:
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(signum)


if __name__ == "__main__":
    (ROOT / ".runtime").mkdir(exist_ok=True)
    with (ROOT / ".runtime/study.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another study command is already using this package") from None
        main()
