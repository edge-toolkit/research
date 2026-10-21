"""Run allowlisted reliability benchmarks through a small local browser interface."""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import signal
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from et_experiments.catalog import load_catalog, parameter_definitions
from et_experiments.recovery.core import summarize_runs
from et_experiments.serialization import json_dumps, json_safe
from et_experiments.tracking.o2 import publish_report


class ConfigurationError(ValueError):
    """A benchmark catalog or browser request is invalid."""


@dataclass
class RunState:
    """Mutable state for one adapter process."""

    run_id: str
    scenario: str
    parameters: dict[str, Any]
    status: str = "queued"
    return_code: int | None = None
    logs: list[str] = field(default_factory=list)
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    results: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        """Return state safe to send to the browser."""
        return {
            key: getattr(self, key)
            for key in ("run_id", "scenario", "parameters", "status", "return_code", "logs", "results", "summary")
        }


class BenchmarkController:
    """Validate requests and supervise one process per benchmark run."""

    def __init__(self, catalog_path: Path, results_dir: Path | None = None) -> None:
        """Validate the trusted catalog and resolve output storage without starting adapters."""
        self.catalog_path = catalog_path.resolve()
        self.project_root = self.catalog_path.parent
        self.catalog = load_catalog(self.catalog_path)
        self._validate_catalog()
        self.results_root = (
            results_dir.resolve()
            if results_dir is not None
            else self._inside_project(self.catalog.get("results_dir", "results/controller"))
        )
        self.runs: dict[str, RunState] = {}
        self.lock = threading.RLock()
        self.workers: list[threading.Thread] = []
        self.closing = False

    def _validate_catalog(self) -> None:
        """Require allowlisted adapter commands and project-contained catalog paths before execution."""
        if self.catalog.get("schema_version") != 1:
            raise ConfigurationError("catalog schema_version must be 1")
        adapter = self.catalog.get("adapter")
        if not isinstance(adapter, dict) or not adapter.get("command"):
            raise ConfigurationError("catalog adapter.command must be a non-empty list")
        command = adapter["command"]
        if not isinstance(command, list) or not all(isinstance(item, str) and item for item in command):
            raise ConfigurationError("catalog adapter.command must contain non-empty strings")
        if not isinstance(self.catalog.get("scenarios"), dict) or not self.catalog["scenarios"]:
            raise ConfigurationError("catalog scenarios must be a non-empty object")
        self._inside_project(adapter.get("cwd", "."))
        self._inside_project(self.catalog.get("results_dir", "results/controller"))

    def _inside_project(self, relative: str) -> Path:
        """Resolve a catalog-relative path and reject locations outside its project, including symlink escapes."""
        if not isinstance(relative, str):
            raise ConfigurationError("catalog paths must be strings")
        resolved = (self.project_root / relative).resolve()
        if not resolved.is_relative_to(self.project_root):
            raise ConfigurationError(f"catalog path escapes its project: {relative}")
        return resolved

    def public_catalog(self) -> dict[str, Any]:
        """Hide the trusted adapter command while exposing GUI choices."""
        scenarios = {
            name: {**scenario, "parameters": parameter_definitions(self.catalog, name)}
            for name, scenario in self.catalog["scenarios"].items()
        }
        return {
            "name": self.catalog.get("name", self.project_root.name),
            "description": self.catalog.get("description", ""),
            "scenarios": scenarios,
        }

    def validate_request(self, value: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Resolve one browser request against its allowlisted scenario schema."""
        scenario_name = value.get("scenario")
        if not isinstance(scenario_name, str):
            raise ConfigurationError("scenario must be a string")
        scenario = self.catalog["scenarios"].get(scenario_name)
        if not isinstance(scenario, dict):
            raise ConfigurationError(f"unknown scenario: {scenario_name!r}")
        supplied = value.get("parameters", {})
        if not isinstance(supplied, dict):
            raise ConfigurationError("parameters must be an object")
        definitions = parameter_definitions(self.catalog, scenario_name)
        unknown = sorted(set(supplied) - set(definitions))
        if unknown:
            raise ConfigurationError(f"unsupported parameters: {', '.join(unknown)}")
        parameters = {
            name: self._validate_parameter(name, supplied.get(name, definition.get("default")), definition)
            for name, definition in definitions.items()
        }
        return str(scenario_name), parameters

    @staticmethod
    def _validate_parameter(name: str, value: Any, definition: dict[str, Any]) -> Any:
        """Enforce the catalog's declared type, finite numeric bounds and permitted choices."""
        kind = definition.get("type")
        valid = {
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "string": isinstance(value, str),
            "boolean": isinstance(value, bool),
        }
        if kind not in valid or not valid[kind]:
            raise ConfigurationError(f"parameter {name!r} must be {kind}")
        if kind in {"integer", "number"}:
            if isinstance(value, float) and not math.isfinite(value):
                raise ConfigurationError(f"parameter {name!r} must be finite")
            if "minimum" in definition and value < definition["minimum"]:
                raise ConfigurationError(f"parameter {name!r} is below its minimum")
            if "maximum" in definition and value > definition["maximum"]:
                raise ConfigurationError(f"parameter {name!r} is above its maximum")
        choices = definition.get("choices")
        if choices is not None and value not in choices:
            raise ConfigurationError(f"parameter {name!r} is not an allowed choice")
        return value

    def start(self, request: dict[str, Any]) -> RunState:
        """Create a run and start its trusted adapter in the background."""
        scenario, parameters = self.validate_request(request)
        run_id = f"benchmark-{uuid.uuid4().hex[:12]}"
        state = RunState(run_id, scenario, parameters)
        with self.lock:
            if self.closing:
                raise ConfigurationError("the controller is shutting down")
            if any(run.status in {"queued", "running", "cancelling"} for run in self.runs.values()):
                raise ConfigurationError("an experiment is already active; wait for cleanup before starting another")
            self.runs[run_id] = state
            thread = threading.Thread(target=self._run, args=(state,), name=run_id)
            self.workers.append(thread)
            thread.start()
        return state

    def _run(self, state: RunState) -> None:
        """Run sequential adapter repetitions and publish terminal state after completion or failure.

        Keep cancellation active until adapter exit, retain each result and stop a live adapter if processing fails.
        """
        adapter = self.catalog["adapter"]
        run_dir = self.results_root / state.run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            repetitions = int(state.parameters.get("repetitions", 1))
            for repetition in range(1, repetitions + 1):
                with self.lock:
                    if state.status in {"cancelling", "cancelled"}:
                        break
                    state.status = "running"
                    state.logs.append(f"Repetition {repetition}/{repetitions}")
                request_file = run_dir / f"request-{repetition}.json"
                result_file = run_dir / f"result-{repetition}.json"
                request_file.write_text(
                    json_dumps(
                        {
                            "run_id": state.run_id,
                            "scenario": state.scenario,
                            "parameters": state.parameters,
                            "repetition": repetition,
                            "project": self.catalog.get("name", self.project_root.name),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                environment = os.environ.copy()
                environment["BENCHMARK_REQUEST_FILE"] = str(request_file)
                environment["BENCHMARK_RESULT_FILE"] = str(result_file)
                environment["BENCHMARK_OUTPUT_DIR"] = str(run_dir)
                # Export here once, with catalog identity and repetition labels.
                environment["BENCHMARK_O2_DEFER"] = "1"
                process = subprocess.Popen(
                    adapter["command"],
                    cwd=self._inside_project(adapter.get("cwd", ".")),
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    start_new_session=os.name != "nt",
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                )
                with self.lock:
                    state.process = process
                    if state.status == "cancelling":
                        self._interrupt(state)
                if process.stdout is not None:
                    for line in process.stdout:
                        with self.lock:
                            state.logs = (state.logs + [line.rstrip()])[-2000:]
                    process.stdout.close()
                return_code = process.wait()
                result = (
                    json.loads(result_file.read_text())
                    if result_file.exists()
                    else {
                        "status": "failed",
                        "error": "adapter did not provide a result",
                    }
                )
                if return_code != 0:
                    result["status"] = "failed"
                    result["adapter_exit_code"] = return_code
                evidence = result
                try:
                    report_path = Path(result.get("report", "")).resolve()
                    if report_path.is_relative_to(run_dir.resolve()) and report_path.is_file():
                        evidence = {**json.loads(report_path.read_text()), "status": result.get("status", "failed")}
                except (OSError, ValueError, TypeError):
                    # Accept a minimal adapter result as well as a detailed report.
                    evidence = result
                if os.environ.get("BENCHMARK_O2_URL"):
                    with self.lock:
                        state.logs.append("Exporting results to OpenObserve (best effort)")
                result["o2"] = publish_report(
                    evidence,
                    run_dir,
                    {
                        "application": self.catalog.get("name", self.project_root.name),
                        "controller_run_id": state.run_id,
                        "job_id": f"{state.run_id}-{repetition}",
                        "scenario": state.scenario,
                        "repetition": repetition,
                    },
                )
                result_file.write_text(json_dumps(result, indent=2) + "\n", encoding="utf-8")
                with self.lock:
                    state.results.append(result)
                    state.return_code = return_code
                    state.summary = json_safe(summarize_runs(state.results))
                (run_dir / "summary.json").write_text(json_dumps(state.summary, indent=2) + "\n")
            with self.lock:
                if state.status == "cancelling":
                    state.status = "cancelled"
                else:
                    state.status = "passed" if all(r.get("status") == "passed" for r in state.results) else "failed"
        # The worker must release the active-run guard even for unexpected adapter or summary errors.
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                state.logs.append(f"adapter failed: {type(exc).__name__}: {exc}")
            try:
                if state.process is not None and state.process.poll() is None:
                    state.process.terminate()
                    try:
                        state.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        state.process.kill()
                        state.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired) as stop_error:
                with self.lock:
                    state.logs.append(f"adapter termination failed: {type(stop_error).__name__}: {stop_error}")
            with self.lock:
                state.status = "failed"
                state.return_code = 127

    def get(self, run_id: str) -> RunState | None:
        """Return a run by identifier."""
        with self.lock:
            return self.runs.get(run_id)

    def cancel(self, run_id: str) -> RunState:
        """Request adapter interruption while retaining the active-run guard through cleanup."""
        with self.lock:
            state = self.runs.get(run_id)
            if state is None:
                raise ConfigurationError("unknown run")
            if state.status in {"queued", "running"}:
                state.status = "cancelling"
                self._interrupt(state)
            return state

    def _interrupt(self, state: RunState) -> None:
        """Signal the isolated adapter process without closing its output pipe or killing its cleanup."""
        process = state.process
        if process is None or process.poll() is not None:
            return
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        except ProcessLookupError:
            # The adapter can exit between poll() and signal delivery.
            pass

    def shutdown(self) -> None:
        """Stop accepting runs and wait for adapter cleanup while workers continue draining stdout."""
        with self.lock:
            self.closing = True
            for state in self.runs.values():
                if state.status in {"queued", "running"}:
                    self.cancel(state.run_id)
            workers = list(self.workers)
        for worker in workers:
            worker.join()


def handler_for(controller: BenchmarkController, index: bytes):
    """Create an HTTP handler bound to one controller instance."""

    class Handler(BaseHTTPRequestHandler):
        def _validate_request(self, *, post: bool = False) -> bool:
            """Restrict browser access to the bound interface and same-origin JSON requests."""
            port = self.server.server_address[1]
            hosts = {self.server.server_address[0], self.connection.getsockname()[0]}
            addresses = {ipaddress.ip_address(host) for host in hosts}
            names = {str(address) for address in addresses if not address.is_unspecified}
            if any(address.is_loopback for address in addresses):
                names.add("localhost")
            authorities = {f"[{name}]:{port}" if ":" in name else f"{name}:{port}" for name in names}
            if port == 80:
                authorities.update(names)
            host = self.headers.get("Host", "").lower()
            if len(self.headers.get_all("Host", [])) != 1 or host not in authorities:
                self._json(HTTPStatus.FORBIDDEN, {"error": "Host must match the controller address"})
                return False
            origins = self.headers.get_all("Origin", [])
            if origins and (len(origins) != 1 or origins[0].lower() != f"http://{host}"):
                self._json(HTTPStatus.FORBIDDEN, {"error": "Origin must match the controller address"})
                return False
            if post and (
                len(self.headers.get_all("Content-Type", [])) != 1
                or self.headers.get_content_type() != "application/json"
            ):
                self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Content-Type must be application/json"})
                return False
            return True

        def _json(self, status: HTTPStatus, value: Any) -> None:
            """Send a JSON response with an explicit content type and byte length."""
            payload = json_dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _request_json(self) -> dict[str, Any]:
            """Read the bounded request body and require a JSON object for run parameters."""
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0:
                raise ConfigurationError("Content-Length must be nonnegative")
            length = min(length, 64 * 1024)
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ConfigurationError("request body must be an object")
            return value

        def do_GET(self) -> None:
            """Validate the request authority before serving UI assets, catalog data or run state."""
            if not self._validate_request():
                return
            path = urlparse(self.path).path
            if path == "/":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(index)))
                self.end_headers()
                self.wfile.write(index)
                return
            if path == "/api/catalog":
                self._json(HTTPStatus.OK, controller.public_catalog())
                return
            if path.startswith("/api/runs/"):
                state = controller.get(path.rsplit("/", 1)[-1])
                self._json(HTTPStatus.OK if state else HTTPStatus.NOT_FOUND, state.public() if state else {})
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            """Validate origin and JSON content type before starting or cancelling an experiment."""
            if not self._validate_request(post=True):
                return
            path = urlparse(self.path).path
            try:
                if path == "/api/runs":
                    self._json(HTTPStatus.ACCEPTED, controller.start(self._request_json()).public())
                    return
                if path.startswith("/api/runs/") and path.endswith("/cancel"):
                    run_id = path.split("/")[-2]
                    self._json(HTTPStatus.OK, controller.cancel(run_id).public())
                    return
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except (ConfigurationError, json.JSONDecodeError, ValueError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def log_message(self, format_string: str, *args: Any) -> None:
            """Prefix HTTP access messages so they can be identified in controller console output."""
            print(f"benchmark-ui: {format_string % args}")

    return Handler


def main() -> int:
    """Parse the CLI and serve the controller UI on the selected interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument(
        "--results-dir", type=Path, help="Operator-selected output root, independent of the application"
    )
    parser.add_argument("command", choices=("serve", "list"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()
    controller = BenchmarkController(args.catalog, args.results_dir)
    if args.command == "list":
        for name, scenario in controller.catalog["scenarios"].items():
            print(f"{name:28} {scenario.get('description', '')}")
        return 0
    index = Path(__file__).with_name("index.html").read_bytes()
    server = ThreadingHTTPServer((args.host, args.port), handler_for(controller, index))
    print(f"Reliability Benchmark Controller: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            controller.shutdown()
        finally:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
