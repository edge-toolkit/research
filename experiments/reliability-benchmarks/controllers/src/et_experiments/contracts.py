"""Application boundaries for the experiment engine."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Protocol

from et_experiments.backends.kubernetes import KubernetesBackend
from et_experiments.tracking.mlflow import TrackingConfig


@dataclass(frozen=True)
class WorkloadConfig:
    context: str
    namespace: str
    primary: str
    workers: tuple[str, ...]
    supporting: tuple[str, ...] = ()
    stress_image: str = ""

    @property
    def workloads(self) -> tuple[str, ...]:
        return (*self.workers, self.primary)

    @property
    def observed(self) -> tuple[str, ...]:
        return (*self.workloads, *self.supporting)


class ApplicationAdapter(Protocol):
    config: WorkloadConfig
    tracking: TrackingConfig | None

    def recovery_config(self) -> dict[str, Any]: ...
    def prepare_scenario(self, scenario: dict[str, Any]) -> dict[str, Any]: ...
    def expected_units(self, scenario: dict[str, Any]) -> int: ...
    def observe(
        self, logs: dict[str, str], tracking_state: dict[str, Any] | None, expected_units: int, started: dt.datetime
    ) -> dict[str, Any]: ...
    def enrich_report(self, report: dict[str, Any]) -> dict[str, Any]: ...
    def tracking_recovery(self, state, expected_units: int, fault, verified_at=None) -> dict[str, Any]: ...
    def configure_run(
        self, backend: KubernetesBackend, faults: Any, job_id: str, scenario: dict[str, Any], started: dt.datetime
    ) -> None: ...
    def restore(self, backend: KubernetesBackend) -> None: ...
    def application_fault(self, fault: dict[str, Any]) -> dict[str, Any] | None: ...
