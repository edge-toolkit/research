"""CLI shared by application-owned experiment entry points."""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

from et_experiments.utils import BenchmarkError

from .engine import BenchmarkEngine


def main(engine: BenchmarkEngine, default_scenarios: Path, default_output: Path) -> int:
    def cancelled(_signum, _frame):
        raise BenchmarkError("experiment cancelled; restoring configuration")

    signal.signal(signal.SIGTERM, cancelled)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", nargs="?", help="Scenario name from scenarios.json")
    parser.add_argument("--scenarios", type=Path, default=default_scenarios)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--keep-configuration", action="store_true", help="Do not restore benchmark env changes")
    parser.add_argument("--list", action="store_true", help="List scenarios and exit")
    args = parser.parse_args()
    scenarios = json.loads(args.scenarios.read_text(encoding="utf-8"))
    if args.list:
        for name, value in scenarios.items():
            print(f"{name:24} {value.get('description', '')}")
        return 0
    if args.scenario not in scenarios:
        raise BenchmarkError(f"unknown scenario {args.scenario!r}; choose from {', '.join(scenarios)}")
    report = engine.execute(
        args.scenario, scenarios[args.scenario], args.output, args.sample_interval, args.keep_configuration
    )
    output_file = args.output / f"{report['job_id']}.json"
    print(json.dumps({"job_id": report["job_id"], "status": report["status"], "output": str(output_file)}, indent=2))
    return 0 if report["status"] == "passed" else 1
