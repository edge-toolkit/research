#!/usr/bin/env python3
"""Run Harbour-FL experiments using the shared experiment controllers."""

import subprocess
import sys
from pathlib import Path

from et_experiments.benchmarking.cli import main
from et_experiments.benchmarking.engine import BenchmarkEngine
from et_experiments.utils import BenchmarkError
from integration import HarbourFLAdapter

if __name__ == "__main__":
    try:
        raise SystemExit(
            main(
                BenchmarkEngine(HarbourFLAdapter()),
                Path(__file__).with_name("scenarios.json"),
                Path(__file__).parents[1] / "results" / "benchmarks",
            )
        )
    except (BenchmarkError, subprocess.TimeoutExpired) as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
