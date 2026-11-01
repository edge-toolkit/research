"""Policy-driven orchestration demonstration for the harbour FL scenario.

The generic policy and runtime boundary are intentionally independent of the
Harbour-FL adapter so they can later move into an Edge Toolkit service.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments.scheduling import placement
from experiments.scheduling.placement import (
    KubernetesRuntime,
    LeastAllocatedPolicy,
    OrchestrationError,
    PlacementDecision,
)

ROUND_COMPLETE = re.compile(r"coordinator completed round (\d+):")
_logger = logging.getLogger(__name__)

NodeState = placement.NodeState
RuntimeAdapter = placement.RuntimeAdapter
WorkloadState = placement.WorkloadState
parse_cpu = placement.parse_cpu
parse_memory = placement.parse_memory


class HarbourFLAdapter:
    CLIENTS = ("vessel-1", "vessel-2", "vessel-3")
    COORDINATOR = "vessel-coordinator"
    SUPPORTING = ("ws-server", "mlflow", "mlflow-bridge")

    def __init__(self, runtime: KubernetesRuntime) -> None:
        self.runtime = runtime

    def prepare(self, job_id: str, rounds: int, timeout_seconds: int) -> None:
        """Start supporting services and assign matching application/recovery identities before launching FL roles."""
        for deployment in self.SUPPORTING:
            self.runtime.scale(deployment, 1)
        for deployment in ("ws-server", "mlflow"):
            self.runtime.wait_rollout(deployment, timeout_seconds)
        for deployment in (*self.CLIENTS, self.COORDINATOR):
            self.runtime.scale(deployment, 0)
        common = [f"HARBOUR_FL_JOB_ID={job_id}", f"RECOVERY_JOB_ID={job_id}"]
        for deployment in self.CLIENTS:
            self.runtime.set_env(deployment, common)
        self.runtime.set_env(
            self.COORDINATOR,
            common
            + [
                "HARBOUR_FL_AUTO_START=true",
                "HARBOUR_FL_START_DELAY_SECONDS=8",
                f"HARBOUR_FL_ROUNDS={rounds}",
                f"HARBOUR_FL_ROUND_TIMEOUT_SECONDS={timeout_seconds}",
                f"HARBOUR_FL_MLFLOW_RUN_NAME={job_id}",
            ],
        )
        for deployment in self.CLIENTS:
            self.runtime.scale(deployment, 1)
        for deployment in self.CLIENTS:
            self.runtime.wait_rollout(deployment, timeout_seconds)
        self.runtime.scale(self.COORDINATOR, 1)
        self.runtime.wait_rollout(self.COORDINATOR, timeout_seconds)
        self.runtime.wait_rollout("mlflow-bridge", timeout_seconds)

    def completed_rounds(self, since_time: str) -> int:
        """Count distinct logical round completions in coordinator logs since the observation boundary."""
        matches = {
            int(match.group(1)) for match in ROUND_COMPLETE.finditer(self.runtime.logs(self.COORDINATOR, since_time))
        }
        return len(matches)

    def restore_manual_start(self) -> None:
        """Restore the configured manual job identities and remove experiment-specific auto-start settings."""
        recovery = json.loads((Path(__file__).resolve().parents[2] / "recovery.json").read_text())
        common = [f"HARBOUR_FL_JOB_ID={recovery['job_id']}", f"RECOVERY_JOB_ID={recovery['job_id']}"]
        self.runtime.set_env(
            self.COORDINATOR,
            [
                "HARBOUR_FL_AUTO_START-",
                "HARBOUR_FL_START_DELAY_SECONDS-",
                "HARBOUR_FL_ROUNDS-",
                "HARBOUR_FL_ROUND_TIMEOUT_SECONDS-",
                "HARBOUR_FL_MLFLOW_RUN_NAME-",
            ]
            + common,
        )
        for deployment in self.CLIENTS:
            self.runtime.set_env(deployment, common)


def now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def iso(value: dt.datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def progress(started: dt.datetime, message: str) -> None:
    elapsed = int((now() - started).total_seconds())
    minutes, seconds = divmod(elapsed, 60)
    print(f"[{minutes:02d}:{seconds:02d}] {message}", flush=True)


def wait_for_round(
    adapter: HarbourFLAdapter, since_time: str, target: int, deadline: float, started: dt.datetime
) -> int:
    """Poll committed-round logs until the target is reached or the observation deadline expires."""
    last = -1
    while time.monotonic() < deadline:
        completed = adapter.completed_rounds(since_time)
        if completed != last:
            progress(started, f"Observed {completed} completed FL rounds")
            last = completed
        if completed >= target:
            return completed
        time.sleep(2)
    raise OrchestrationError(f"timed out waiting for {target} completed rounds")


def validate_policy(policy_data: dict[str, Any]) -> None:
    """Reject unsupported targets, round counts, trigger points and action modes before changing Kubernetes."""
    workload_id = str(policy_data.get("workload_id", "vessel-2"))
    rounds = int(policy_data.get("rounds", 5))
    trigger = int(policy_data.get("trigger_after_round", 0))
    action_mode = str(policy_data.get("action_mode", "auto"))
    if workload_id not in HarbourFLAdapter.CLIENTS:
        raise OrchestrationError(f"orchestration target must be one of {HarbourFLAdapter.CLIENTS!r}")
    if not 3 <= rounds <= 20:
        raise OrchestrationError("rounds must be between 3 and 20")
    if trigger < 0 or trigger >= rounds - 1:
        raise OrchestrationError("trigger_after_round must leave at least one recovery round")
    if action_mode not in {"auto", "baseline", "restart", "relocate", "expect-no-target"}:
        raise OrchestrationError("action_mode must be auto, baseline, restart, relocate, or expect-no-target")


def execute(policy_data: dict[str, Any], output: Path, keep_placement: bool) -> dict[str, Any]:
    """Observe training, apply a placement decision and record progress, restoring manual settings during cleanup."""
    validate_policy(policy_data)
    started = now()
    context = str(policy_data.get("context", "k3d-harbour-fl"))
    namespace = str(policy_data.get("namespace", "et-harbour-fl"))
    workload_id = str(policy_data.get("workload_id", "vessel-2"))
    rounds = int(policy_data.get("rounds", 5))
    trigger_after_round = int(policy_data.get("trigger_after_round", 0))
    timeout_seconds = int(policy_data.get("timeout_seconds", 600))
    rollout_timeout = int(policy_data.get("rollout_timeout_seconds", 180))
    action_mode = str(policy_data.get("action_mode", "auto"))
    runtime = KubernetesRuntime(context, namespace)
    adapter = HarbourFLAdapter(runtime)
    policy = LeastAllocatedPolicy(
        {str(key): str(value) for key, value in policy_data.get("required_node_labels", {}).items()},
        bool(policy_data.get("allow_same_node_restart", True)),
    )
    job_id = f"orchestrated-harbour-fl-{started.strftime('%Y%m%d%H%M%S%f')}"
    report: dict[str, Any] = {
        "schema_version": 1,
        "job_id": job_id,
        "scenario": str(policy_data.get("scenario", action_mode)),
        "action_mode": action_mode,
        "started_at": iso(started),
        "status": "running",
        "events": [],
    }
    original_placement: dict[str, str] | None = None
    action_attempted = False
    observation_start = iso(started)
    try:
        progress(started, f"Preparing {rounds}-round Harbour-FL run {job_id}")
        adapter.prepare(job_id, rounds, rollout_timeout)
        if action_mode == "baseline":
            wait_for_round(adapter, observation_start, rounds, time.monotonic() + timeout_seconds, started)
            report.update(
                {
                    "status": "passed",
                    "finished_at": iso(now()),
                    "decision": None,
                    "metrics": {"decision_latency_seconds": 0.0, "rounds_lost": 0},
                }
            )
            progress(started, "No-action scheduling baseline completed")
            return report
        wait_for_round(adapter, observation_start, trigger_after_round + 1, time.monotonic() + timeout_seconds, started)

        observed_at = now()
        workload = runtime.workload(workload_id)
        nodes = runtime.nodes()
        original_placement = runtime.placement(workload_id)
        decision_started = time.monotonic()
        try:
            if action_mode == "restart":
                decision = PlacementDecision(
                    workload_id,
                    "restart",
                    workload.node_name,
                    workload.node_name,
                    "scenario requires a controlled same-node recovery",
                    (workload.node_name,),
                )
            else:
                selected_policy = policy
                if action_mode in {"relocate", "expect-no-target"}:
                    selected_policy = LeastAllocatedPolicy(policy.required_labels, False)
                decision = selected_policy.decide(workload, nodes)
        except OrchestrationError as exc:
            decision_seconds = time.monotonic() - decision_started
            if action_mode == "relocate":
                report.update(
                    {
                        "status": "skipped",
                        "finished_at": iso(now()),
                        "skip_reason": str(exc),
                        "nodes": [asdict(node) for node in nodes],
                        "metrics": {"decision_latency_seconds": decision_seconds},
                    }
                )
                progress(started, f"Strict relocation skipped: {exc}")
                return report
            if action_mode == "expect-no-target":
                report.update(
                    {
                        "status": "passed",
                        "finished_at": iso(now()),
                        "expected_rejection_observed": True,
                        "rejection_reason": str(exc),
                        "nodes": [asdict(node) for node in nodes],
                        "metrics": {"decision_latency_seconds": decision_seconds, "rounds_lost": 0},
                    }
                )
                progress(started, f"Expected policy rejection observed: {exc}")
                return report
            raise
        decision_seconds = time.monotonic() - decision_started
        if action_mode == "expect-no-target":
            raise OrchestrationError("expect-no-target scenario unexpectedly found an eligible node")
        progress(
            started, f"Decision: {decision.action} {workload_id} from {decision.source_node} to {decision.target_node}"
        )
        report["events"].append({"at": iso(observed_at), "type": "decision", **asdict(decision)})

        action_started = now()
        action_attempted = True
        if decision.action == "relocate":
            recovered = runtime.relocate(workload_id, decision.target_node, rollout_timeout)
        else:
            recovered = runtime.restart(workload_id, rollout_timeout)
        ready_at = now()
        progress(started, f"Replacement pod {recovered.pod_name} is ready on {recovered.node_name}")
        report["events"].append({"at": iso(ready_at), "type": "workload-ready", **asdict(recovered)})

        completed = wait_for_round(
            adapter, observation_start, trigger_after_round + 2, time.monotonic() + timeout_seconds, started
        )
        resumed_at = now()
        wait_for_round(adapter, observation_start, rounds, time.monotonic() + timeout_seconds, started)
        report.update(
            {
                "status": "passed",
                "finished_at": iso(now()),
                "decision": asdict(decision),
                "nodes": [asdict(node) for node in nodes],
                "metrics": {
                    "decision_latency_seconds": decision_seconds,
                    "replacement_ready_seconds": (ready_at - action_started).total_seconds(),
                    "resume_seconds": (resumed_at - action_started).total_seconds(),
                    "completed_rounds_at_recovery": completed,
                    "rounds_lost": 0,
                },
            }
        )
        progress(started, "Orchestration demonstration completed")
        return report
    except Exception as exc:
        report.update({"status": "failed", "finished_at": iso(now()), "error": str(exc)})
        raise
    finally:
        cleanup_errors = []
        cleanup_actions = [("manual settings", adapter.restore_manual_start)]
        if action_attempted and not keep_placement:
            cleanup_actions.insert(
                0, ("placement", lambda: runtime.restore_placement(workload_id, original_placement, rollout_timeout))
            )
        for step, action in cleanup_actions:
            try:
                action()
            except Exception as error:  # Cleanup must retain the operation's error and still finalize its report.
                _logger.exception("Scheduling cleanup failed for %s", step)
                cleanup_errors.append({"step": step, "error": str(error), "error_type": type(error).__name__})
        if cleanup_errors:
            report["cleanup_errors"] = cleanup_errors
            report["status"] = "failed"
            report.setdefault("error", "Cleanup failed; inspect cleanup_errors before another experiment")
        report.setdefault("finished_at", iso(now()))
        report["wall_time_seconds"] = (now() - started).total_seconds()
        output.mkdir(parents=True, exist_ok=True)
        destination = output / f"{job_id}.json"
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Report: {destination}", flush=True)


def main() -> int:
    """Validate the selected policy, optionally print its plan, or execute it and return a CLI status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path(__file__).with_name("policy.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--keep-placement", action="store_true")
    parser.add_argument(
        "--show-plan", action="store_true", help="Print policy configuration without changing Kubernetes"
    )
    args = parser.parse_args()
    policy_data = json.loads(args.policy.read_text(encoding="utf-8"))
    validate_policy(policy_data)
    if args.show_plan:
        print(json.dumps(policy_data, indent=2, sort_keys=True))
        return 0
    try:
        report = execute(policy_data, args.output, args.keep_placement)
    except (OrchestrationError, subprocess.TimeoutExpired, ValueError) as exc:
        print(f"orchestration error: {exc}", flush=True)
        return 1
    print(json.dumps({"job_id": report["job_id"], "status": report["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
