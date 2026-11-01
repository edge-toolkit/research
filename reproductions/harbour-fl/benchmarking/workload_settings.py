"""Validate pretrained-workload settings and derive per-experiment Core policy values."""

from __future__ import annotations

from et_experiments.utils import BenchmarkError
from harbour_fl.workload import workload


def policy_settings(scenario, defaults):
    """Keep the stall policy longer than a configured round collection deadline."""
    try:
        workload(scenario.get("model_profile", "pretrained"))
    except ValueError as error:
        raise BenchmarkError(str(error)) from error
    overrides = scenario.get("recovery_policy", {})
    if not isinstance(overrides, dict) or not set(overrides).issubset(defaults):
        raise BenchmarkError("recovery_policy must contain known Core policy settings")
    policy = dict(defaults)
    timeout = scenario.get("round_timeout_seconds", 180)
    if type(timeout) is not int or not 1 <= timeout <= 3600:
        raise BenchmarkError("round_timeout_seconds must be an integer between 1 and 3600")
    policy["stall_seconds"] = max(policy["stall_seconds"], timeout + 60)
    policy.update(overrides)
    for key, value in policy.items():
        minimum, maximum = (0, 5) if key == "max_restarts" else (1, 7200)
        if type(value) is not int or not minimum <= value <= maximum:
            raise BenchmarkError(f"{key} must be an integer between {minimum} and {maximum}")
    if policy["deadline_seconds"] <= policy["detection_grace_seconds"] + policy["restart_after_seconds"]:
        raise BenchmarkError("recovery deadline must permit detection and restart")
    return policy


def policy_environment(policy):
    """Use Core's existing serde-env names for the scenario-specific policy."""
    return [f"RECOVERY_POLICY_{key.upper()}={value}" for key, value in policy.items()]


def resilience_settings(scenario, defaults):
    """Permit bounded timing overrides while keeping the deployment's dependency allowlist fixed."""
    overrides = scenario.get("recovery_resilience", {})
    if not isinstance(overrides, dict) or not set(overrides).issubset(defaults):
        raise BenchmarkError("recovery_resilience must contain known Core resilience settings")
    settings = {**defaults, **overrides}
    if settings.get("dependencies", {}) != defaults.get("dependencies", {}):
        raise BenchmarkError("dependency changes require scenario regeneration with matching read-only permissions")
    for name, value in settings.items():
        if name == "dependencies":
            continue
        if name == "reconnect":
            if type(value) is not bool:
                raise BenchmarkError("reconnect must be boolean")
            continue
        lower = 1 if name in {"connect_timeout_seconds", "backoff_multiplier", "backoff_max_seconds"} else 0
        upper = 16 if name == "backoff_multiplier" else 7200
        if type(value) is not int or not lower <= value <= upper:
            raise BenchmarkError(f"{name} must be an integer between {lower} and {upper}")
    if settings.get("jitter_seconds", 0) > settings.get("backoff_max_seconds", 300):
        raise BenchmarkError("jitter_seconds cannot exceed backoff_max_seconds")
    return settings
