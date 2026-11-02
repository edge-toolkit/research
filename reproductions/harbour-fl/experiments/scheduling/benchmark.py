"""Run repeatable dynamic-scheduling scenarios against Harbour-FL on k3s."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from experiments.scheduling import controller as CONTROLLER


def merged_policy(base: dict[str, Any], scenario_name: str, scenario: dict[str, Any]) -> dict[str, Any]:
    """Apply scenario overrides to the base policy without treating descriptive text as a setting."""
    result = dict(base)
    result.update({key: value for key, value in scenario.items() if key != "description"})
    result["scenario"] = scenario_name
    return result


def summary_row(report: dict[str, Any], repetition: int) -> dict[str, Any]:
    """Flatten the available decision, timing and failure evidence into one CSV row."""
    metrics = report.get("metrics", {})
    decision = report.get("decision") or {}
    return {
        "scenario": report.get("scenario"),
        "repetition": repetition,
        "job_id": report.get("job_id"),
        "status": report.get("status"),
        "action": decision.get("action"),
        "source_node": decision.get("source_node"),
        "target_node": decision.get("target_node"),
        "wall_time_seconds": report.get("wall_time_seconds"),
        "decision_latency_seconds": metrics.get("decision_latency_seconds"),
        "replacement_ready_seconds": metrics.get("replacement_ready_seconds"),
        "resume_seconds": metrics.get("resume_seconds"),
        "rounds_lost": metrics.get("rounds_lost"),
        "skip_reason": report.get("skip_reason"),
        "error": report.get("error"),
    }


def append_summary(path: Path, row: dict[str, Any]) -> None:
    """Append a scheduling result, writing the CSV header only when the file is created."""
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    """Run selected scenarios and repetitions, retaining failure rows while continuing after expected errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", nargs="?", default="")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--scenarios", type=Path, default=Path(__file__).with_name("scenarios.json"))
    parser.add_argument("--policy", type=Path, default=Path(__file__).with_name("policy.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args()

    scenarios = json.loads(args.scenarios.read_text(encoding="utf-8"))
    if args.list:
        for name, scenario in scenarios.items():
            print(f"{name:24} {scenario['description']}")
        return 0
    names = list(scenarios) if args.all else [args.scenario]
    if not args.all and args.scenario not in scenarios:
        print(f"unknown scenario {args.scenario!r}; use --list", file=sys.stderr)
        return 2
    if not 1 <= args.repeat <= 20:
        print("--repeat must be between 1 and 20", file=sys.stderr)
        return 2

    base = json.loads(args.policy.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    failed = False
    for name in names:
        for repetition in range(1, args.repeat + 1):
            print(f"\n=== {name} repetition {repetition}/{args.repeat} ===", flush=True)
            policy = merged_policy(base, name, scenarios[name])
            try:
                report = CONTROLLER.execute(policy, args.output, keep_placement=False)
            except (CONTROLLER.OrchestrationError, subprocess.TimeoutExpired, ValueError) as exc:
                failed = True
                row = {
                    "scenario": name,
                    "repetition": repetition,
                    "job_id": None,
                    "status": "failed",
                    "action": None,
                    "source_node": None,
                    "target_node": None,
                    "wall_time_seconds": None,
                    "decision_latency_seconds": None,
                    "replacement_ready_seconds": None,
                    "resume_seconds": None,
                    "rounds_lost": None,
                    "skip_reason": None,
                    "error": str(exc),
                }
            else:
                row = summary_row(report, repetition)
                failed = failed or report.get("status") == "failed"
            append_summary(args.output / "scheduling-summary.csv", row)
            print(json.dumps(row, indent=2), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
