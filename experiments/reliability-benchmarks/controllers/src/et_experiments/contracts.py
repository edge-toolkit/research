"""Application boundaries for the experiment engine."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Protocol

from et_experiments.backends.kubernetes import KubernetesBackend
from et_experiments.tracking.mlflow import TrackingConfig


@dataclass(frozen=True)
class WorkloadConfig:
    """Identify the primary, worker and supporting deployments within one Kubernetes context and namespace."""

    context: str
    namespace: str
    primary: str
    workers: tuple[str, ...]
    supporting: tuple[str, ...] = ()
    stress_image: str = ""

    @property
    def workloads(self) -> tuple[str, ...]:
        """Return workers and the primary deployment subject to application recovery."""
        return (*self.workers, self.primary)

    @property
    def observed(self) -> tuple[str, ...]:
        """Include supporting services in resource and log observations."""
        return (*self.workloads, *self.supporting)


class ApplicationAdapter(Protocol):
    """Connect application-owned deployment, progress and fault semantics to the generic engine."""

    config: WorkloadConfig
    tracking: TrackingConfig | None

    def recovery_config(self) -> dict[str, Any]:
        """Supply supervisor deployment, workload allowlist and default recovery settings."""
        ...

    def prepare_scenario(self, scenario: dict[str, Any]) -> dict[str, Any]:
        """Translate application options into engine settings before deployment changes."""
        ...

    def expected_units(self, scenario: dict[str, Any]) -> int:
        """Return the positive count of successful work units required to complete this scenario."""
        ...

    def observe(
        self, logs: dict[str, str], tracking_state: dict[str, Any] | None, expected_units: int, started: dt.datetime
    ) -> dict[str, Any]:
        """Reconcile application evidence into completed units, durations, final metrics and failure state.

        Use started to scope observations to this run; optional tracking data may be unavailable.
        """
        ...

    def enrich_report(self, report: dict[str, Any]) -> dict[str, Any]:
        """Add application measurements to the report and return numeric SLIs for SLO evaluation."""
        ...

    def tracking_recovery(self, state, expected_units: int, fault, verified_at=None) -> dict[str, Any]:
        """Assess whether expected application results reached tracking after the configured disruption."""
        ...

    def configure_run(
        self, backend: KubernetesBackend, faults: Any, job_id: str, scenario: dict[str, Any], started: dt.datetime
    ) -> None:
        """Prepare fault helpers and application settings, then start the job identified by job_id."""
        ...

    def restore(self, backend: KubernetesBackend) -> None:
        """Restore application configuration after an experiment, including interrupted runs."""
        ...

    def application_fault(self, fault: dict[str, Any]) -> dict[str, Any] | None:
        """Apply a domain-specific fault, returning evidence or None when the type is unsupported."""
        ...
