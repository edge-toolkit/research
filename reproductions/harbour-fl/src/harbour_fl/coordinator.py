"""Federated coordinator role and update validation."""

from __future__ import annotations

import json
import logging
import os
import pickle
import queue
import time
from typing import Any

from .checkpoints import _checkpoint_bytes, _load_checkpoint
from .data import _data_root, _labels, _load_rows
from .devices import select_device
from .instrumentation import checkpoint_key
from .learning import _evaluate, _fed_avg, _import_torch, _new_model
from .quality import evaluate_inputs
from .settings import (
    COMPLETE_TYPE,
    DEFAULT_CLIENTS,
    ROUND_TYPE,
    SCHEMA_VERSION,
    UPDATE_TYPE,
    _bool_env,
    _positive_int_env,
)
from .vision import preprocessing_metadata, preprocessing_signature
from .workload import describe

_logger = logging.getLogger("harbour_fl")


class CoordinatorRole:
    def _coordinator_main(self) -> None:
        """Start or resume federated rounds, commit recovery checkpoints and publish final evaluation and tracking
        events."""
        job_id = os.environ.get("HARBOUR_FL_JOB_ID", "harbour-fl-demo")
        try:
            pointer = self.state["storage"].get(self.state["agent_id"], f"{checkpoint_key(job_id)}-latest.json")
            interrupted = pointer is not None and json.loads(bytes(pointer))["progress"] < _positive_int_env(
                "HARBOUR_FL_ROUNDS", 3
            )
            if not interrupted or not _bool_env("HARBOUR_FL_RESUME", True):
                self._mark_ready()
            resuming = interrupted and _bool_env("HARBOUR_FL_RESUME", True)
            if resuming:
                self.state["started"] = True
                self.state["start"].set()
                self._event("resume-started")
            if _bool_env("HARBOUR_FL_AUTO_START") and (not resuming):
                time.sleep(float(os.environ.get("HARBOUR_FL_START_DELAY_SECONDS", "3")))
                with self.state["start_lock"]:
                    self.state["started"] = True
                    self.state["start"].set()
                _logger.info("coordinator auto-start enabled")
            elif not resuming:
                _logger.info("coordinator waiting for a browser start command for job %s", job_id)
                self._send_coordinator_status("waiting-for-start", expected_clients=len(DEFAULT_CLIENTS))
            while not self.state["start"].wait(timeout=0.5):
                if self.state["stop"].is_set():
                    return
            if self.state["stop"].is_set():
                return
            self._send_status(job_id, "running")
            self._send_coordinator_status("run-started")
            torch = _import_torch()
            torch.manual_seed(_positive_int_env("HARBOUR_FL_TRAINING_SEED", 17))
            torch.set_num_threads(_positive_int_env("HARBOUR_FL_TORCH_THREADS", 2))
            root = _data_root()
            labels = _labels(root)
            test_rows = _load_rows(root, root / "test_manifest.csv")
            model = _new_model(torch, len(labels))
            device = select_device(torch)
            image_preprocessing = preprocessing_signature()
            self._event("image-preprocessing", settings=preprocessing_metadata())
            model.to(device)
            self._event("workload-profile", settings={**describe(model, torch, device), "test_samples": len(test_rows)})
            clients = tuple(
                item.strip() for item in os.environ.get("HARBOUR_FL_CLIENTS", ",".join(DEFAULT_CLIENTS)).split(",")
            )
            rounds = _positive_int_env("HARBOUR_FL_ROUNDS", 3)
            checkpoints = self.state["events"].checkpoints(self.state["storage"], self.state["agent_id"])
            resume_started = time.monotonic()
            saved = checkpoints.load() if _bool_env("HARBOUR_FL_RESUME", True) else None
            history = []
            start_round = 0
            if saved is not None:
                recovered = _load_checkpoint(saved[0], torch)
                if recovered.get("image_preprocessing", {"enabled": False}) != image_preprocessing:
                    raise ValueError("saved checkpoint uses different image preprocessing")
                if (
                    recovered.get("job_id"),
                    recovered["labels"],
                    recovered.get("rounds"),
                    recovered.get("clients"),
                ) != (job_id, labels, rounds, list(clients)):
                    raise ValueError("saved run does not match the configured job, labels, rounds, or clients")
                model.load_state_dict(recovered["state_dict"], strict=True)
                torch.set_rng_state(recovered["torch_rng"])
                start_round = recovered["round"]
                history = recovered["history"]
                self._event(
                    "resume-completed",
                    progress=start_round,
                    seconds=time.monotonic() - resume_started,
                    repeated_work_units=None,
                )
                for record in history:
                    _logger.info("coordinator completed round %d: %s", record["round"], record["metrics"])
                    self._emit_mlflow(
                        "metrics",
                        f"{job_id}:round:{record['round']}",
                        job_id,
                        step=record["round"],
                        metrics=record["mlflow_metrics"],
                    )
            else:
                self._save_recovery_checkpoint(
                    checkpoints,
                    _checkpoint_bytes(
                        model,
                        labels,
                        0,
                        torch,
                        job_id=job_id,
                        rounds=rounds,
                        clients=list(clients),
                        history=[],
                        torch_rng=torch.get_rng_state(),
                        image_preprocessing=image_preprocessing,
                    ),
                    0,
                )
            self._emit_mlflow(
                "run-start",
                f"{job_id}:start",
                job_id,
                params={
                    "clients": len(clients),
                    "rounds": rounds,
                    "local_epochs": _positive_int_env("HARBOUR_FL_LOCAL_EPOCHS", 1),
                    "aggregation": "fedavg",
                    "modalities": "log_mel,camera_image",
                    "classes": len(labels),
                    "adaptation": _bool_env("HARBOUR_FL_ADAPTATION"),
                    "minimum_clients": _positive_int_env("HARBOUR_FL_MIN_CLIENTS", len(clients)),
                    "coordinator_device": str(device),
                    "image_preprocessing": json.dumps(preprocessing_metadata(), sort_keys=True),
                },
                tags={"application": "vessel-classification", "runner": "pyo3"},
            )
            self._mark_ready()
            evaluation = _evaluate(model, test_rows, labels, torch) if start_round == rounds else None
            for round_number in range(start_round, rounds):
                round_started = time.monotonic()
                self._event("operation-started", progress=round_number)
                self._send_coordinator_status(
                    "round-started", round=round_number, expected_clients=len(clients), received_clients=[]
                )
                checkpoint_name = f"{job_id}-global-round-{round_number:03d}.pt"
                fail_round = int(os.environ.get("HARBOUR_FL_FAIL_CHECKPOINT_WRITE_ROUND", "-1"))
                if round_number == fail_round:
                    raise OSError(f"injected checkpoint write failure for round {round_number}")
                checkpoint_started = time.monotonic()
                self.state["storage"].put(
                    checkpoint_name,
                    _checkpoint_bytes(model, labels, round_number, torch, image_preprocessing=image_preprocessing),
                )
                _logger.info(
                    "coordinator checkpoint write round %d completed in %.6f seconds",
                    round_number,
                    time.monotonic() - checkpoint_started,
                )
                updates = self._collect_round(job_id, round_number, checkpoint_name, clients)
                self._send_coordinator_status(
                    "aggregating",
                    round=round_number,
                    expected_clients=len(clients),
                    received_clients=[update["client_id"] for update in updates],
                )
                _fed_avg(model, updates, torch)
                self._send_coordinator_status("evaluating", round=round_number)
                evaluation = _evaluate(model, test_rows, labels, torch)
                confusion_matrix = evaluation.pop("confusion_matrix")
                client_accuracy = sum(update["metrics"]["validation_accuracy"] for update in updates) / len(updates)
                metrics = {
                    "global_test_loss": evaluation["loss"],
                    "global_test_accuracy": evaluation["accuracy"],
                    "global_test_macro_f1": evaluation["macro_f1"],
                    "mean_client_validation_accuracy": client_accuracy,
                    "participating_clients": float(len(updates)),
                    "round_work_seconds": time.monotonic() - round_started,
                }
                history.append(
                    {
                        "round": round_number,
                        "metrics": metrics,
                        "mlflow_metrics": self._mlflow_round_metrics(metrics, updates),
                    }
                )
                if (round_number + 1) % _positive_int_env(
                    "HARBOUR_FL_CHECKPOINT_INTERVAL", 1
                ) == 0 or round_number + 1 == rounds:
                    self._save_recovery_checkpoint(
                        checkpoints,
                        _checkpoint_bytes(
                            model,
                            labels,
                            round_number + 1,
                            torch,
                            job_id=job_id,
                            rounds=rounds,
                            clients=list(clients),
                            history=history,
                            torch_rng=torch.get_rng_state(),
                            image_preprocessing=image_preprocessing,
                        ),
                        round_number + 1,
                    )
                self._event("progress", progress=round_number + 1, metrics=metrics)
                self._emit_mlflow(
                    "metrics",
                    f"{job_id}:round:{round_number}",
                    job_id,
                    step=round_number,
                    metrics=self._mlflow_round_metrics(metrics, updates),
                )
                confusion_artifacts = self._confusion_matrix_artifacts(job_id, round_number, labels, confusion_matrix)
                self._emit_mlflow(
                    "artifacts", f"{job_id}:confusion:{round_number}", job_id, artifacts=confusion_artifacts
                )
                self._send_coordinator_status("round-completed", round=round_number, metrics=metrics)
                _logger.info("coordinator completed round %d: %s", round_number, metrics)
            if start_round >= rounds:
                # A replacement can restore final weights while final evaluation/artifact delivery is unfinished.
                evaluation = _evaluate(model, test_rows, labels, torch)
            robustness = (
                evaluate_inputs(
                    model,
                    test_rows,
                    labels,
                    torch,
                    _evaluate(model, test_rows, labels, torch, input_condition={}),
                    self._event,
                    rounds,
                )
                if _bool_env("HARBOUR_FL_EVALUATE_DEGRADED_INPUTS")
                else None
            )
            if robustness is not None:
                self._emit_mlflow(
                    "metrics",
                    f"{job_id}:robustness",
                    job_id,
                    step=rounds,
                    metrics={
                        f"prediction_{condition}_{key}": values[key]
                        for condition, values in robustness.items()
                        for key in ("accuracy", "macro_f1")
                    },
                )
            final_name = f"{job_id}-final-model.pt"
            self.state["storage"].put(
                final_name, _checkpoint_bytes(model, labels, rounds, torch, image_preprocessing=image_preprocessing)
            )
            summary_name = f"{job_id}-summary.json"
            summary = {
                "job_id": job_id,
                "rounds": rounds,
                "clients": clients,
                "labels": labels,
                "test_samples": len(test_rows),
                "image_preprocessing": image_preprocessing,
                "metrics": evaluation,
                "prediction_input_evaluations": robustness,
            }
            self.state["storage"].put(summary_name, json.dumps(summary, indent=2, sort_keys=True).encode())
            artifacts = [
                {"bucket": self.state["agent_id"], "filename": final_name, "artifact_path": "model/final-model.pt"},
                {
                    "bucket": self.state["agent_id"],
                    "filename": summary_name,
                    "artifact_path": "evaluation/summary.json",
                },
            ]
            self._emit_mlflow("artifacts", f"{job_id}:artifacts", job_id, artifacts=artifacts)
            self._emit_mlflow("run-end", f"{job_id}:finish", job_id, step=rounds, status="FINISHED")
            self._send_json(
                {
                    "type": COMPLETE_TYPE,
                    "schema_version": SCHEMA_VERSION,
                    "job_id": job_id,
                    "bucket": self.state["agent_id"],
                    "filename": final_name,
                    "summary": summary,
                }
            )
            self._send_status(job_id, "completed", metrics=evaluation)
            self._send_coordinator_status("run-completed", metrics=evaluation)
            self._event("completed", progress=rounds, metrics=evaluation)
            self.state["completed"] = True
        except Exception:  # Final run boundary: publish failure evidence; Core owns operational recovery.
            self._mark_failed()
            self._event("failed")
            _logger.exception("harbour FL coordinator failed")
            self._send_status(job_id, "failed")
            self._send_coordinator_status("failed")
            self._emit_mlflow("run-end", f"{job_id}:failed", job_id, status="FAILED")

    def _save_recovery_checkpoint(self, checkpoints, payload: bytes, progress: int) -> None:
        """Expose a pre-commit boundary and an optional bounded write-delay experiment."""
        self._event("checkpoint-save-started", progress=progress, bytes=len(payload))
        delay = float(os.environ.get("HARBOUR_FL_CHECKPOINT_WRITE_DELAY_SECONDS", "0"))
        if not 0 <= delay <= 30:
            raise ValueError("checkpoint write delay must be between 0 and 30 seconds")
        if delay:
            time.sleep(delay)
        checkpoints.save(payload, progress)

    def _collect_round(self, job_id: str, round_number: int, checkpoint_name: str, clients: tuple[str, ...]):
        """Collect validated client updates until all arrive or the deadline permits a configured reduced quorum."""
        deadline = time.monotonic() + float(os.environ.get("HARBOUR_FL_ROUND_TIMEOUT_SECONDS", "180"))
        updates: dict[str, dict[str, Any]] = {}
        last_broadcast = 0.0
        while len(updates) < len(clients):
            now = time.monotonic()
            if now >= deadline:
                missing = sorted(set(clients) - set(updates))
                minimum = _positive_int_env("HARBOUR_FL_MIN_CLIENTS", len(clients))
                if minimum <= len(updates):
                    self._event("degraded-participation", missing=missing, received=len(updates), required=minimum)
                    return [updates[client] for client in clients if client in updates]
                raise TimeoutError(f"round {round_number} timed out waiting for {missing}")
            if now - last_broadcast >= 2.0:
                self._send_json(
                    {
                        "type": ROUND_TYPE,
                        "schema_version": SCHEMA_VERSION,
                        "job_id": job_id,
                        "round": round_number,
                        "coordinator_id": self.state["agent_id"],
                        "filename": checkpoint_name,
                    }
                )
                last_broadcast = now
            try:
                message = self.state["update_queue"].get(timeout=min(0.5, deadline - now))
            except queue.Empty:
                continue
            if message is None:
                raise RuntimeError("coordinator stopped")
            client_id = message.get("client_id")
            if (
                message.get("type") != UPDATE_TYPE
                or message.get("job_id") != job_id
                or message.get("round") != round_number
                or (client_id not in clients)
            ):
                continue
            if message.get("status") != "ready":
                if message.get("recoverable") is True:
                    self._event("client-recovery-pending", participant=client_id, progress=round_number)
                    continue  # Retain the existing deadline while Core replaces the failed client.
                raise RuntimeError(f"{client_id} failed round {round_number}: {message.get('error', 'unknown error')}")
            if client_id in updates:
                continue
            payload = self.state["storage"].get(message["bucket"], message["filename"])
            if payload is None:
                continue
            torch = _import_torch()
            try:
                candidate = _load_checkpoint(bytes(payload), torch)
                reference = _load_checkpoint(
                    bytes(self.state["storage"].get(self.state["agent_id"], checkpoint_name)), torch
                )
                if candidate.get("image_preprocessing", {"enabled": False}) != reference.get(
                    "image_preprocessing", {"enabled": False}
                ):
                    raise ValueError("client update uses different image preprocessing")
                if candidate["labels"] != reference["labels"] or candidate["round"] != round_number:
                    raise ValueError("update labels or round do not match")
                samples = message.get("samples")
                if type(samples) is not int or not 1 <= samples <= 10000000:
                    raise ValueError("invalid update sample count")
                if candidate["state_dict"].keys() != reference["state_dict"].keys():
                    raise ValueError("update parameters do not match")
                for name, tensor in candidate["state_dict"].items():
                    expected_tensor = reference["state_dict"][name]
                    if (
                        not torch.is_tensor(tensor)
                        or tensor.shape != expected_tensor.shape
                        or tensor.dtype != expected_tensor.dtype
                        or (not bool(torch.isfinite(tensor).all()))
                    ):
                        raise ValueError("invalid update shape or nonfinite parameter")
                    if tensor.is_floating_point() and float(torch.linalg.vector_norm(tensor - expected_tensor)) > float(
                        os.environ.get("HARBOUR_FL_MAX_UPDATE_NORM", "1000")
                    ):
                        raise ValueError("update exceeds the configured parameter-delta norm bound")
                message["checkpoint"] = candidate
            except (ValueError, TypeError, RuntimeError, KeyError, pickle.UnpicklingError, EOFError) as exc:
                self._event("update-rejected", participant=client_id, reason=str(exc)[:200], progress=round_number)
                continue
            updates[client_id] = message
            self._event(
                "contribution-accepted",
                participant=client_id,
                progress=round_number + 1,
                participant_instance=message.get("participant_instance"),
            )
            self._send_json(
                {
                    "type": "harbour-fl-contribution-accepted",
                    "schema_version": SCHEMA_VERSION,
                    "job_id": job_id,
                    "client_id": client_id,
                    "round": round_number,
                }
            )
            self._send_coordinator_status(
                "waiting-for-clients",
                round=round_number,
                expected_clients=len(clients),
                received_clients=sorted(updates),
            )
        return [updates[client] for client in clients]
