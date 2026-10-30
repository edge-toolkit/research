"""Reproducible fault triggers and bounded combined-fault designs."""

from __future__ import annotations

import math
import random
import sys

from et_experiments.utils import BenchmarkError


def validate_design(scenario, workers):
    """Validate experimental controls before deployment mutation."""
    timeout = scenario.get("timeout_seconds", 900)
    if type(timeout) not in (int, float) or not 0 < timeout <= sys.float_info.max:
        raise BenchmarkError("timeout_seconds must be a finite positive number")
    seed = scenario.get("seed", 17)
    if "post_fault_progress_timeout_seconds" in scenario:
        timeout = scenario["post_fault_progress_timeout_seconds"]
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 1 <= timeout <= 7200:
            raise BenchmarkError("post_fault_progress_timeout_seconds must be finite and between 1 and 7200")
    if type(scenario.get("require_recovery_stability", False)) is not bool:
        raise BenchmarkError("require_recovery_stability must be boolean")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise BenchmarkError("seed must be an integer between 0 and 2**32-1")
    if not isinstance(scenario.get("reference", {}), dict):
        raise BenchmarkError("reference must be an object")
    fault = scenario.get("fault", {})
    if fault.get("type") == "combined":
        faults = fault.get("faults", [])
        permitted = {"temporary_scale_down", "network_latency", "network_bandwidth", "network_loss", "degraded_network"}
        if not isinstance(faults, list) or len(faults) != 2:
            raise BenchmarkError("combined requires exactly two bounded client faults")
        for child in faults:
            if child.get("type") not in permitted or child.get("target") not in workers:
                raise BenchmarkError("combined faults require supported types and allowlisted workers")
            seconds = child.get("duration_seconds", 15)
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 1 <= seconds <= 300:
                raise BenchmarkError("combined fault durations must be between 1 and 300 seconds")
        if len({child["target"] for child in faults}) != 2:
            raise BenchmarkError("combined faults must target different workers")
    return seed


class FaultTrigger:
    """Trigger from observed progress or diagnostic events, with recorded seeded delay."""

    def __init__(self, fault, seed=17):
        """Validate trigger bounds and resolve seeded stage selection and delay once per experiment."""
        rng = random.Random(seed)
        trigger = fault.get("trigger", {})
        if not isinstance(trigger, dict):
            raise BenchmarkError("fault trigger must be an object")
        choices = trigger.get("choices")
        if choices is not None:
            if not isinstance(choices, list) or not choices or len(choices) > 16:
                raise BenchmarkError("trigger choices must contain 1 to 16 objects")
            trigger = rng.choice(choices)
        if not isinstance(trigger, dict):
            raise BenchmarkError("fault trigger must be an object")
        self.settings = dict(trigger)
        self.settings.setdefault("after_work_unit", fault.get("after_work_unit", -1))
        for key, maximum in (("after_seconds", 14400), ("delay_seconds", 300), ("jitter_seconds", 60)):
            value = trigger.get(key, 0)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
                raise BenchmarkError(f"trigger {key} must be between 0 and {maximum}")
        if type(self.settings["after_work_unit"]) is not int or self.settings["after_work_unit"] < -1:
            raise BenchmarkError("trigger after_work_unit must be an integer >= -1")
        if "event_kind" in trigger and (not isinstance(trigger["event_kind"], str) or not trigger["event_kind"]):
            raise BenchmarkError("event_kind must be a nonempty diagnostic event name")
        if "progress" in trigger and (type(trigger["progress"]) is not int or trigger["progress"] < 0):
            raise BenchmarkError("trigger progress must be a nonnegative integer")
        self.delay = trigger.get("delay_seconds", 0) + rng.uniform(0, trigger.get("jitter_seconds", 0))
        self.evidence = None

    def ready(self, completed, elapsed, events=()):
        """Latch the first matching observation and report readiness after the configured delay.

        Work-unit and elapsed-time gates must also pass. Event progress is filtered only when explicitly configured.
        """
        if completed <= self.settings["after_work_unit"] or elapsed < self.settings.get("after_seconds", 0):
            return False
        matching = [
            e
            for e in events
            if e.get("kind") == self.settings.get("event_kind")
            and (not self.settings.get("workload") or e.get("workload") == self.settings["workload"])
            and ("progress" not in self.settings or e.get("progress", -1) >= self.settings["progress"])
        ]
        if self.settings.get("event_kind") and not matching and self.evidence is None:
            return False
        if self.evidence is None:
            self.evidence = {
                "observed_elapsed_seconds": elapsed,
                "completed_units": completed,
                "event_id": matching[0]["id"] if matching else None,
                "event_at": matching[0].get("at") if matching else None,
                "event_progress": matching[0].get("progress") if matching else None,
            }
        return elapsed - self.evidence["observed_elapsed_seconds"] >= self.delay

    def report(self):
        """Return the selected trigger, sampled delay and first-observation evidence for the report."""
        return {
            "selected": self.settings,
            "delay_seconds": self.delay,
            "evidence": self.evidence,
            "basis": "controller-observed trigger; injection may occur after the application stage has advanced",
        }
