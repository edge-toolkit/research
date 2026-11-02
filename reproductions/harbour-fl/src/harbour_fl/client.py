"""Client-side training role."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from .checkpoints import _checkpoint_bytes, _load_checkpoint
from .data import _data_root, _labels, _load_rows
from .devices import gpu_error, publish_gpu_state, select_device
from .learning import _evaluate, _import_torch, _new_model, _train_local
from .settings import SCHEMA_VERSION, UPDATE_TYPE, _bool_env, _positive_int_env
from .transfers import measure_transfer
from .vision import preprocessing_metadata, preprocessing_signature
from .workload import describe

_logger = logging.getLogger("harbour_fl")


class ClientRole:
    def _client_main(self) -> None:
        """Load client data and process round requests, reporting failures and stopping after fatal accelerator
        errors."""
        torch = _import_torch()
        torch.manual_seed(_positive_int_env("HARBOUR_FL_TRAINING_SEED", 17))
        torch.set_num_threads(_positive_int_env("HARBOUR_FL_TORCH_THREADS", 2))
        self.state["device"] = select_device(torch)
        publish_gpu_state(self.state["device"])
        self._event("compute-device", device=str(self.state["device"]))
        self._event("image-preprocessing", settings=preprocessing_metadata())
        root = _data_root()
        client_id = self.state["client_id"]
        labels = _labels(root)
        train_rows = _load_rows(root, root / "clients" / client_id / "train_manifest.csv")
        val_rows = _load_rows(root, root / "clients" / client_id / "val_manifest.csv")
        _logger.info("%s loaded %d train and %d validation pairs", client_id, len(train_rows), len(val_rows))
        self._mark_ready()
        self._send_client_status("dataset-loaded", train_samples=len(train_rows), validation_samples=len(val_rows))
        while not self.state["stop"].is_set():
            message = self.state["update_queue"].get()
            if message is None:
                return
            try:
                self._run_client_round(message, train_rows, val_rows, labels, torch)
            except Exception as exc:  # Report the local round failure without killing the connected agent.
                accelerator_failed = gpu_error(exc, self.state["device"])
                self.state["seen_rounds"].discard(message.get("round"))
                _logger.exception("%s failed round %s", client_id, message.get("round"))
                self._send_client_status("failed", round=message.get("round"), error=str(exc)[:500])
                self._send_json(
                    {
                        "type": UPDATE_TYPE,
                        "schema_version": SCHEMA_VERSION,
                        "job_id": message.get("job_id"),
                        "round": message.get("round"),
                        "client_id": client_id,
                        "status": "failed",
                        "error": str(exc)[:500],
                        "recoverable": accelerator_failed,
                    }
                )
                if accelerator_failed:
                    self.state["gpu_failed"] = True
                    readiness = os.environ.get("HARBOUR_FL_READINESS_FILE")
                    if readiness:
                        Path(readiness).unlink(missing_ok=True)
                    return  # Core must replace the worker before its GPU context is used again.

    def _run_client_round(self, message, train_rows, val_rows, labels, torch) -> None:
        """Validate and download the global head, train locally, then upload and cache the client contribution."""
        expected_job = os.environ.get("HARBOUR_FL_JOB_ID", "harbour-fl-demo")
        if message.get("job_id") != expected_job:
            raise ValueError("unexpected FL job")
        coordinator = message.get("coordinator_id")
        filename = message.get("filename")
        if not isinstance(coordinator, str) or not isinstance(filename, str):
            raise TypeError("round message has no checkpoint pointer")
        payload, download_seconds = measure_transfer(
            self, "download", message.get("round"), lambda: self.state["storage"].get(coordinator, filename)
        )
        checkpoint = _load_checkpoint(bytes(payload), torch)
        if checkpoint.get("image_preprocessing", {"enabled": False}) != preprocessing_signature():
            raise ValueError("coordinator and client image preprocessing do not match")
        if checkpoint["labels"] != labels or checkpoint["round"] != message["round"]:
            raise ValueError("global checkpoint metadata does not match this client")
        model = _new_model(torch, len(labels))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(self.state.get("device") or select_device(torch))
        self._event(
            "workload-profile",
            settings={
                **describe(model, torch, next(model.parameters()).device),
                "train_samples": len(train_rows),
                "validation_samples": len(val_rows),
            },
        )
        torch.manual_seed(_positive_int_env("HARBOUR_FL_TRAINING_SEED", 17) + int(message["round"]))
        self._send_client_status(
            "training", round=message["round"], train_samples=len(train_rows), validation_samples=len(val_rows)
        )
        self._event("training-started", progress=message["round"])
        train_metrics = _train_local(model, train_rows, labels, message["round"], torch)
        validation = _evaluate(model, val_rows, labels, torch)
        update_name = f"{message['job_id']}-round-{message['round']:03d}-{self.state['client_id']}.pt"
        self._send_client_status("uploading-update", round=message["round"])
        self._event("update-transfer-started", progress=message["round"])
        serialization_started = time.monotonic()
        update_payload = _checkpoint_bytes(
            model, labels, message["round"], torch, image_preprocessing=preprocessing_signature()
        )
        serialization_seconds = time.monotonic() - serialization_started
        _, upload_seconds = measure_transfer(
            self,
            "upload",
            message["round"],
            lambda: self.state["storage"].put(update_name, update_payload),
            payload_bytes=len(update_payload),
        )
        if _bool_env("HARBOUR_FL_INVALID_UPDATE"):
            for parameter in model.parameters():
                parameter.data.fill_(float("nan"))
            invalid_payload = _checkpoint_bytes(
                model, labels, message["round"], torch, image_preprocessing=preprocessing_signature()
            )
            measure_transfer(
                self,
                "upload",
                message["round"],
                lambda: self.state["storage"].put(update_name, invalid_payload),
                payload_bytes=len(invalid_payload),
            )
        metrics = {
            **train_metrics,
            "global_download_seconds": download_seconds,
            "global_checkpoint_bytes": len(payload),
            "update_bytes": len(update_payload),
            "update_serialization_seconds": serialization_seconds,
            "update_upload_seconds": upload_seconds,
            "gpu_enabled": float(next(model.parameters()).device.type != "cpu"),
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
        }
        response = {
            "type": UPDATE_TYPE,
            "schema_version": SCHEMA_VERSION,
            "job_id": message["job_id"],
            "round": message["round"],
            "client_id": self.state["client_id"],
            "participant_instance": getattr(self.state.get("events"), "instance", None),
            "status": "ready",
            "bucket": self.state["agent_id"],
            "filename": update_name,
            "samples": len(train_rows),
            "metrics": metrics,
        }
        self.state["responses"][message["round"]] = response
        self._event("contribution-ready", progress=message["round"] + 1)
        self._send_json(response)
        self._send_client_status("waiting", round=message["round"], samples=len(train_rows), metrics=metrics)
        _logger.info("%s completed round %d: %s", self.state["client_id"], message["round"], metrics)
