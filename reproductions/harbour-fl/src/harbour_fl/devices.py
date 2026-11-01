"""Compute selection and opt-in application-boundary accelerator fault injection."""

from __future__ import annotations

import json
import os
from pathlib import Path

GPU_FAULT_ROOT = Path("/tmp/et-benchmark-gpu")


class GPUExecutionError(RuntimeError):
    """The accelerator execution context must be replaced, not silently retried on CPU."""


def select_device(torch, requested=None):
    """Keep CPU defaults; explicit accelerator requests fail closed when unavailable."""
    requested = (requested or os.environ.get("HARBOUR_FL_DEVICE", "cpu")).strip().lower()
    cuda = torch.cuda.is_available()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "auto":
        requested = "cuda" if cuda else "mps" if mps else "cpu"
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("HARBOUR_FL_DEVICE must be cpu, cuda, mps, or auto")
    if (requested == "cuda" and not cuda) or (requested == "mps" and not mps):
        raise RuntimeError(f"requested {requested} accelerator is unavailable; CPU fallback is disabled")
    return torch.device(requested)


def synchronize(torch, device):
    """Include asynchronous accelerator work in elapsed training measurements."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def gpu_error(error, device):
    """Treat accelerator execution failures as fatal to this worker, not ordinary invalid training data."""
    if isinstance(error, GPUExecutionError):
        return True
    return (
        device.type in {"cuda", "mps"}
        and isinstance(error, RuntimeError)
        and any(
            text in str(error).lower() for text in ("cuda", "cudnn", "mps", "metal", "out of memory", "device-side")
        )
    )


def _write(path, document):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document))
    temporary.replace(path)


def publish_gpu_state(device):
    """Advertise the selected device only when this workload explicitly opts into benchmark injection."""
    if os.environ.get("ET_BENCHMARK_GPU_FAULTS") != "1":
        return
    GPU_FAULT_ROOT.mkdir(parents=True, exist_ok=True)
    _write(
        GPU_FAULT_ROOT / "state.json",
        {"schema_version": 1, "job_id": os.environ.get("HARBOUR_FL_JOB_ID", "harbour-fl-demo"), "device": device.type},
    )


def check_gpu_fault(device, torch):
    """Consume one job-scoped request and acknowledge it before raising a simulated GPU exception."""
    if os.environ.get("ET_BENCHMARK_GPU_FAULTS") != "1":
        return
    request_path = GPU_FAULT_ROOT / "request.json"
    try:
        request = json.loads(request_path.read_text())
    except FileNotFoundError:
        return
    if request.get("schema_version") != 1 or not isinstance(request.get("request_id"), str):
        raise ValueError("invalid GPU fault request")
    if request.get("job_id") != os.environ.get("HARBOUR_FL_JOB_ID", "harbour-fl-demo"):
        return  # A request left by an older job must not disrupt this training run.
    if device.type not in {"cuda", "mps"} or request.get("mode") not in {"runtime_error", "out_of_memory"}:
        raise ValueError("GPU fault injection requires a GPU and an allowlisted fault mode")
    request_path.unlink()
    _write(GPU_FAULT_ROOT / "receipt.json", {**request, "device": device.type, "status": "injected"})
    message = f"injected {device.type} {request['mode']} (application-level simulation, not hardware failure)"
    if request["mode"] == "out_of_memory":
        raise torch.OutOfMemoryError(message)
    raise GPUExecutionError(message)
