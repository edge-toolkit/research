#!/usr/bin/env python3
"""Translate a validated browser request into a Harbour-FL experiment."""

import json
import os
import sys
from pathlib import Path

from et_experiments.catalog import effective_scenario, load_catalog


def main() -> int:
    """Translate a validated UI request into a seeded scenario and replace this process with the benchmark CLI."""
    request_path = Path(os.environ["BENCHMARK_REQUEST_FILE"]).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    root = Path(__file__).parents[1]
    catalog = load_catalog(root / "benchmark-catalog.json")
    scenario_name = request["scenario"]
    parameters = request.get("parameters", {})
    scenario = effective_scenario(catalog, scenario_name, parameters)
    scenario["seed"] = (int(scenario.get("seed", 17)) + int(request.get("repetition", 1)) - 1) % 2**32
    effective_path = request_path.with_name("effective-scenario.json")
    effective_path.write_text(json.dumps({scenario_name: scenario}, indent=2) + "\n", encoding="utf-8")
    command = [
        sys.executable,
        str(Path(__file__).with_name("benchmark.py")),
        scenario_name,
        "--scenarios",
        str(effective_path),
        "--output",
        os.environ.get("BENCHMARK_OUTPUT_DIR", str(root / "results" / "benchmarks")),
        "--sample-interval",
        str(parameters.get("sample_interval_seconds", 2.0)),
    ]
    if parameters.get("paired_baseline"):
        command.append("--paired-baseline")
    os.execv(sys.executable, command)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
