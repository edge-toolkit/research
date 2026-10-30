"""Independent functional-check scheduling and availability evidence, separate from recovery supervision."""

from __future__ import annotations

import math
import threading
import time
import uuid

from et_experiments.recovery.statistics import distribution
from et_experiments.utils import BenchmarkError


class ObserverUnavailable(RuntimeError):
    """The observer transport failed; this is not evidence of an application outage."""


class ServiceUnavailable(RuntimeError):
    """A check could not use the service through a functioning observer transport."""


def settings(scenario, worker_count):
    """Resolve opt-in settings and bound probe traffic and pending-check lifetimes."""
    value = scenario.get("service_probes", {})
    if not isinstance(value, dict):
        raise BenchmarkError("service_probes must be an object")
    enabled = value.get("enabled", scenario.get("service_probes_enabled", False))
    if type(enabled) is not bool:
        raise BenchmarkError("service probe enabled must be boolean")
    result = {"enabled": enabled}
    if not enabled:
        return result
    for key, default, low, high in (
        ("interval_seconds", 2, 0.5, 60),
        ("timeout_seconds", 5, 0.1, 60),
    ):
        number = value.get(key, default)
        if type(number) not in (int, float) or not math.isfinite(number) or not low <= number <= high:
            raise BenchmarkError(f"probe {key} must be between {low} and {high}")
        result[key] = float(number)
    required = value.get("required_workers", scenario.get("minimum_clients", worker_count))
    payload = value.get("rtt_payload_bytes", 32)
    if type(required) is not int or not 1 <= required <= worker_count:
        raise BenchmarkError("probe required_workers must fit the configured worker count")
    if type(payload) is not int or not 16 <= payload <= 125:
        raise BenchmarkError("RTT payload must be 16–125 bytes")
    return {**result, "required_workers": required, "rtt_payload_bytes": payload}


def summarize(records, workers, config, start, end):
    """Estimate quorum availability by fixed check windows, preserving incomplete observation bounds."""
    interval = config["interval_seconds"]
    elapsed = max(0, end - start)
    indexed = {(r["cycle"], r["target"]): r for r in records}
    available = unavailable = unknown = 0.0
    groups = {worker: [] for worker in workers}
    for cycle in range(math.ceil(elapsed / interval)):
        width = min(interval, elapsed - cycle * interval)
        successes = missing = 0
        for worker in workers:
            record = indexed.get((cycle, worker), {"status": "unknown", "cycle": cycle, "target": worker})
            groups[worker].append(record)
            successes += record["status"] == "success"
            missing += record["status"] == "unknown"
        if successes >= config["required_workers"]:
            available += width
        elif successes + missing < config["required_workers"]:
            unavailable += width
        else:
            unknown += width
    per_worker = {}
    for worker, rows in groups.items():
        good = sum(r["status"] == "success" for r in rows)
        gaps = sum(r["status"] == "unknown" for r in rows)
        bad = len(rows) - good - gaps
        per_worker[worker] = {
            "eligible_checks": len(rows),
            "successful": good,
            "unsuccessful": bad,
            "unknown": gaps,
            "success_ratio": good / (good + bad) if good + bad else None,
            "observation_coverage": (good + bad) / len(rows) if rows else None,
            "functional_latency_ms": distribution([r.get("elapsed_ms") for r in rows if r["status"] == "success"]),
        }
    return {
        "status": "measured" if elapsed and unknown == 0 else "incomplete",
        "scope": "coordinator task dispatch, bounded worker arithmetic and verified result collection",
        "window_started_at": start,
        "window_finished_at": end,
        "eligible_seconds": elapsed,
        "available_seconds": available,
        "unavailable_seconds": unavailable,
        "unknown_seconds": unknown,
        "availability_ratio": available / elapsed if elapsed and unknown == 0 else None,
        "availability_lower_bound": available / elapsed if elapsed else None,
        "availability_upper_bound": (available + unknown) / elapsed if elapsed else None,
        "observation_coverage": (available + unavailable) / elapsed if elapsed else None,
        "required_workers": config["required_workers"],
        "by_worker": per_worker,
        "time_estimation": "each fixed interval is classified by its check outcomes; final interval is truncated",
    }


class ServiceProbeMonitor:
    """Probe independently of the coordinator process, including while it is restarting."""

    def __init__(self, transport, workers, config):
        self.transport, self.workers, self.config = transport, tuple(workers), config
        self.stop = threading.Event()
        self.thread = None
        self.records = []
        self.started_at = self.finished_at = None
        self.origin = None
        self.error = None

    def start(self):
        """Establish the observer before opening the eligible measurement window."""
        try:
            self.transport.start()
        except BaseException:
            self.transport.close()
            raise
        self.started_at, self.origin = time.time(), time.monotonic()
        self.thread = threading.Thread(target=self._loop, name="functional-probes", daemon=True)
        self.thread.start()

    def finish(self):
        """Stop issuing checks and let existing checks resolve before deployment cleanup."""
        if self.finished_at is None and self.origin is not None:
            self.finished_at = self.started_at + time.monotonic() - self.origin
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=self.config["timeout_seconds"] + 15)
            if self.thread.is_alive():
                self.transport.close()
                raise BenchmarkError("functional probe observer did not finish within its shutdown deadline")
        else:
            self.transport.close()
        result = {"settings": self.config, "records": self.records, "observer_error": self.error}
        if self.started_at is not None:
            result.update(summarize(self.records, self.workers, self.config, self.started_at, self.finished_at))
        else:
            result.update(status="not-started", availability_ratio=None)
        return result

    def _loop(self):
        pending = {}
        cycle = 0

        def complete(identifier, status, now, reason=None):
            record = pending.pop(identifier)
            record.update(status=status, finished_at=self.started_at + now - self.origin)
            if status == "success":
                record["elapsed_ms"] = (now - record.pop("sent_monotonic")) * 1000
            else:
                record.pop("sent_monotonic", None)
            record.pop("deadline", None)
            if reason:
                record["reason"] = reason
            self.records.append(record)

        try:
            while not self.stop.is_set() or pending:
                now = time.monotonic()
                if not self.stop.is_set() and now >= self.origin + cycle * self.config["interval_seconds"]:
                    scheduled = self.origin + cycle * self.config["interval_seconds"]
                    for target in self.workers:
                        now = time.monotonic()
                        identifier = uuid.uuid4().hex
                        pending[identifier] = {
                            "probe_id": identifier,
                            "cycle": cycle,
                            "target": target,
                            "scheduled_at": self.started_at + scheduled - self.origin,
                            "sent_monotonic": now,
                            "deadline": now + self.config["timeout_seconds"],
                        }
                        if self.stop.is_set() or now - scheduled >= self.config["interval_seconds"]:
                            complete(identifier, "unknown", now, "observer scheduling gap")
                            continue
                        try:
                            self.transport.send(identifier, target)
                        except ObserverUnavailable:
                            complete(identifier, "unknown", time.monotonic(), "observer transport unavailable")
                        except ServiceUnavailable:
                            complete(identifier, "failure", time.monotonic(), "service connection unavailable")
                    cycle += 1
                now = time.monotonic()
                for identifier in list(pending):
                    if now >= pending[identifier]["deadline"]:
                        complete(identifier, "timeout", now, "response deadline exceeded")
                try:
                    response = self.transport.receive(0.05)
                except (ObserverUnavailable, ServiceUnavailable) as error:
                    status = "unknown" if isinstance(error, ObserverUnavailable) else "failure"
                    for identifier in list(pending):
                        complete(identifier, status, time.monotonic(), type(error).__name__)
                    self.stop.wait(0.1)
                    continue
                if not response or response.get("probe_id") not in pending:
                    continue  # Duplicates, foreign jobs and late replies cannot create another observation.
                identifier = response["probe_id"]
                now = time.monotonic()
                if now >= pending[identifier]["deadline"]:
                    complete(identifier, "timeout", now, "late response")
                elif (
                    response.get("target") == pending[identifier]["target"]
                    and response.get("status") == "success"
                    and type(response.get("result")) is int
                    and response.get("result") == 42
                ):
                    complete(identifier, "success", now)
                else:
                    complete(identifier, "failure", now, "invalid result or worker unavailable")
        except Exception as error:  # noqa: BLE001 - thread boundary retains unknown observer failures
            self.error = type(error).__name__
            for identifier in list(pending):
                complete(identifier, "unknown", time.monotonic(), "observer failed")
        finally:
            self.transport.close()
