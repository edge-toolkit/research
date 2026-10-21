"""Run repeated experiments with matched fault-free baselines and durable series summaries."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import uuid
from pathlib import Path

from et_experiments.recovery.core import summarize_runs
from et_experiments.serialization import json_dumps
from et_experiments.utils import BenchmarkError

from .reference import compare_baseline
from .reports import save_report


def baseline_scenario(scenario):
    """Copy workload and recovery settings while replacing the fault and clearing fault-specific success gates."""
    baseline = copy.deepcopy(scenario)
    baseline["preparation_fault"] = baseline.get("fault", {"type": "none"})
    baseline["fault"] = {"type": "none"}
    baseline["slos"] = {}
    baseline["allow_run_failure"] = False
    for key in tuple(baseline):
        if key.startswith("require_"):
            baseline[key] = False
    return baseline


def execute_series(
    engine, name, scenario, output, sample_interval=2, repetitions=1, paired=False, *, stop_on_failure=False
):
    """Retain failed repetitions and keep each baseline beside its matched fault run."""
    if type(repetitions) is not int or not 1 <= repetitions <= 100:
        raise BenchmarkError("repetitions must be between 1 and 100")
    seed = scenario.get("seed", 17)
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise BenchmarkError("seed must be an integer between 0 and 2**32-1")
    series_id = "series-" + uuid.uuid4().hex[:12]
    output.mkdir(parents=True, exist_ok=True)
    runs, baselines = [], []

    def run(run_name, settings):
        """Recover an engine-persisted failed report, but stop the series when cleanup failed or no report exists."""
        before = set(output.glob("*.json"))
        try:
            return engine.execute(run_name, settings, output, sample_interval, False)
        except (BenchmarkError, subprocess.TimeoutExpired) as error:
            created = set(output.glob("*.json")) - before
            public_result = os.environ.get("BENCHMARK_RESULT_FILE")
            for path in sorted(created):
                if public_result and path.resolve() == Path(public_result).resolve():
                    continue
                report = json.loads(path.read_text())
                if (
                    report.get("job_id") == path.stem
                    and report.get("scenario") == run_name
                    and report.get("status") == "failed"
                ):
                    if report.get("cleanup", {}).get("status") == "failed":
                        raise BenchmarkError(
                            "series stopped after failed cleanup; inspect and restore the deployment"
                        ) from error
                    return report
            raise

    def publish():
        """Persist every completed member, including a failed baseline with no subsequent fault run."""
        summary = summarize_runs(runs)
        summary["baseline_runs"] = len(baselines)
        summary["failed_baseline_runs"] = sum(r.get("status") != "passed" for r in baselines)
        document = {
            "schema_version": 1,
            "series_id": series_id,
            "runs": runs,
            "baselines": baselines,
            "summary": summary,
            "status": "passed" if all(r.get("status") == "passed" for r in runs + baselines) else "failed",
        }
        (output / f"{series_id}.json").write_text(json_dumps(document, indent=2) + "\n")
        return document, summary

    for repetition in range(repetitions):
        settings = {**copy.deepcopy(scenario), "seed": (seed + repetition) % 2**32}
        baseline = run(name + "-baseline", baseline_scenario(settings)) if paired else None
        if baseline is not None:
            baselines.append(baseline)
            if stop_on_failure and baseline.get("status") != "passed":
                document, summary = publish()
                break
        report = run(name, settings)
        if baseline is not None:
            report["baseline_comparison"] = compare_baseline(report, baseline)
        report["series_id"] = series_id
        report["repetition"] = repetition + 1
        save_report(output, report)
        runs.append(report)
        document, summary = publish()
        if stop_on_failure and report.get("status") != "passed":
            break
    keys = {
        "job_id",
        "series_id",
        "repetition",
        "status",
        "application",
        "scenario",
        "fault",
        "fault_configuration",
        "reference",
        "experiment_seed",
        "resilience",
        "recovery",
        "recovery_stability",
        "slos",
        "sli",
        "attribution",
        "cleanup",
        "useful_recovery",
        "service_probes",
        "communication_probes",
        "transfer_measurements",
        "prediction_input_evaluations",
        "tracking_recovery",
        "baseline_comparison",
        "error",
    }
    public_runs = [{key: value for key, value in r.items() if key in keys} for r in runs]
    last = (runs or baselines)[-1]
    public = {
        **{key: value for key, value in last.items() if key in keys},
        "status": document["status"],
        "run_reports": public_runs,
        "series_summary": summary,
    }
    if os.environ.get("BENCHMARK_RESULT_FILE"):
        os_result = {k: v for k, v in public.items() if k not in {"samples", "events", "diagnostic_events"}}
        os_result["report"] = str(output / f"{last['job_id']}.json")
        Path(os.environ["BENCHMARK_RESULT_FILE"]).write_text(json_dumps(os_result, indent=2) + "\n")
    return document
