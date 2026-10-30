"""CLI shared by application-owned experiment entry points."""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

from et_experiments.serialization import json_dumps
from et_experiments.utils import BenchmarkError

from .engine import BenchmarkEngine
from .series import execute_series


def main(engine: BenchmarkEngine, default_scenarios: Path, default_output: Path) -> int:
    """Run or list application scenarios, supporting sequential repetitions, paired baselines and cleanup on SIGTERM."""

    def cancelled(_signum, _frame):
        """Turn termination or Windows console interruption into the experiment's cleanup path."""
        raise KeyboardInterrupt("experiment cancelled; restoring configuration")

    signal.signal(signal.SIGTERM, cancelled)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, cancelled)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", nargs="?", help="Scenario name from scenarios.json")
    parser.add_argument("--scenarios", type=Path, default=default_scenarios)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--keep-configuration", action="store_true", help="Do not restore benchmark env changes")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop after the first failed baseline or experiment, after cleanup",
    )
    parser.add_argument(
        "--paired-baseline", action="store_true", help="Run a matching fault-free baseline before each run"
    )
    parser.add_argument("--seed", type=int, help="Set the initial fault-timing seed; repetitions increment it")
    parser.add_argument("--list", action="store_true", help="List scenarios and exit")
    args = parser.parse_args()
    scenarios = json.loads(args.scenarios.read_text(encoding="utf-8"))
    if args.list:
        for name, value in scenarios.items():
            print(f"{name:24} {value.get('description', '')}")
        return 0
    if args.scenario not in scenarios:
        raise BenchmarkError(f"unknown scenario {args.scenario!r}; choose from {', '.join(scenarios)}")
    scenario = dict(scenarios[args.scenario])
    if args.seed is not None:
        scenario["seed"] = args.seed
    if args.repetitions != 1 or args.paired_baseline:
        if args.keep_configuration:
            raise BenchmarkError("series require cleanup between runs; omit --keep-configuration")
        document = execute_series(
            engine,
            args.scenario,
            scenario,
            args.output,
            args.sample_interval,
            args.repetitions,
            args.paired_baseline,
            stop_on_failure=args.stop_on_failure,
        )
        print(json_dumps(document["summary"], indent=2))
        return 0 if document["status"] == "passed" else 1
    report = engine.execute(args.scenario, scenario, args.output, args.sample_interval, args.keep_configuration)
    output_file = args.output / f"{report['job_id']}.json"
    print(json_dumps({"job_id": report["job_id"], "status": report["status"], "output": str(output_file)}, indent=2))
    return 0 if report["status"] == "passed" else 1
