"""Describe repeated recovery experiments with explicit populations and uncertainty."""

from __future__ import annotations

import json
import math
import statistics

from et_experiments.recovery.supervision import measurements as supervision_measurements
from et_experiments.utils import percentile


def distribution(values, allow_negative=False):
    """Summarize finite samples, withholding p95 below 20 observations and p99 below 100."""
    samples = [v for v in values if type(v) in (int, float) and math.isfinite(v) and (allow_negative or v >= 0)]
    return {
        "samples": len(samples),
        "mean": statistics.fmean(samples) if samples else None,
        "min": min(samples, default=None),
        "max": max(samples, default=None),
        "p50": percentile(samples, 0.5),
        "p95": percentile(samples, 0.95) if len(samples) >= 20 else None,
        "p99": percentile(samples, 0.99) if len(samples) >= 100 else None,
        "sample_standard_deviation": statistics.stdev(samples) if len(samples) > 1 else None,
    }


def success_interval(successes, total):
    """Return a Wilson 95% interval for a binomial incident success proportion."""
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    scale = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / scale
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / scale
    return {
        "lower": max(0, centre - margin),
        "upper": min(1, centre + margin),
        "confidence": 0.95,
        "method": "Wilson; assumes independent comparable incidents",
    }


def comparable_groups(results):
    """Group like-for-like fault conditions and report recovery distributions, unsuccessful outcomes and uncertainty."""
    groups = {}
    for result in results:
        key = {
            "application": result.get("application", "unspecified"),
            "scenario": result.get("scenario", "unspecified"),
            "reference": result.get("reference", {}).get("fingerprint", "unspecified"),
            "fault_type": result.get("fault", {}).get("type", "unspecified"),
            "target": result.get("fault", {}).get("target"),
            "fault_configuration": result.get("fault_configuration", result.get("effective_scenario", {}).get("fault")),
            "trigger_stage": result.get("fault", {}).get("trigger", {}).get("selected"),
            "recovery_mode": result.get("attribution", {}).get("recovery_mode", "unspecified"),
            "provider": result.get("recovery", {}).get("provider", "unspecified"),
            "recovery_clock_basis": result.get("recovery", {}).get("clock_basis", "unspecified"),
            "recovery_resilience": result.get("recovery", {}).get("resilience", {}),
        }
        group = groups.setdefault(json.dumps(key, sort_keys=True), {"conditions": key, "runs": []})
        group["runs"].append(result)
    summaries = []
    for group in groups.values():
        runs = group["runs"]
        incidents = [
            i
            for r in runs
            for i in r.get("recovery", {}).get("incidents", [])
            if i.get("status") not in {"suspected", "transient", "observing"}
        ]
        recovered = [i for i in incidents if i.get("status") == "recovered"]
        durations = [i["recovered_at"] - i["observed_failure_at"] for i in recovered]
        detection = [i["detected_at"] - i["observed_failure_at"] for i in incidents if i.get("detected_at") is not None]
        measured_slis = {name for run in runs for name in run.get("sli", {})}
        comparisons = [
            r["baseline_comparison"] for r in runs if r.get("baseline_comparison", {}).get("status") == "compared"
        ]
        compared_metrics = {name for comparison in comparisons for name in comparison.get("metrics", {})}
        summaries.append(
            {
                "conditions": group["conditions"],
                "repetitions": len(runs),
                "passed_runs": sum(r.get("status") == "passed" for r in runs),
                "incidents": len(incidents),
                "recovered_incidents": len(recovered),
                "unrecovered_incidents": len(incidents) - len(recovered),
                "runs_without_detected_incidents": sum(
                    not any(
                        i.get("status") not in {"suspected", "transient", "observing"}
                        for i in r.get("recovery", {}).get("incidents", [])
                    )
                    for r in runs
                ),
                "recovery_seconds": distribution(durations),
                "failure_detection_seconds": distribution(detection),
                "recovery_success_ratio": len(recovered) / len(incidents) if incidents else None,
                "recovery_success_interval": success_interval(len(recovered), len(incidents)),
                "supervision": supervision_measurements(
                    incidents, [gap for run in runs for gap in run.get("recovery", {}).get("observation_gaps", [])]
                ),
                "sli_distributions": {
                    name: distribution([r.get("sli", {}).get(name) for r in runs], True)
                    for name in sorted(measured_slis)
                },
                "matched_baseline_runs": len(comparisons),
                "baseline_delta_distributions": {
                    name: distribution([c.get("metrics", {}).get(name, {}).get("delta") for c in comparisons], True)
                    for name in sorted(compared_metrics)
                },
            }
        )
    return summaries
