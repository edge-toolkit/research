"""Simulate application-visible GPU errors through an opt-in, one-shot request protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from et_experiments.utils import BenchmarkError

# Applications opt in by publishing state.json and checking request.json at a safe execution boundary. The receipt
# confirms consumption of this exact request; the benchmark engine separately verifies recovery.
GPU_FAULT_SCRIPT = """
import json, pathlib, sys, time, uuid
root = pathlib.Path('/tmp/et-benchmark-gpu')
job_id, mode, seconds = sys.argv[1:]
state = json.loads((root / 'state.json').read_text())
if state.get('schema_version') != 1 or state.get('job_id') != job_id or state.get('device') not in ('cuda', 'mps'):
    raise RuntimeError('GPU injection requires an opted-in GPU workload for this job; CPU simulation is refused')
if mode not in ('runtime_error', 'out_of_memory'):
    raise ValueError('unsupported GPU fault mode')
request = {'schema_version': 1, 'job_id': job_id, 'mode': mode, 'request_id': uuid.uuid4().hex}
temporary = root / 'request.tmp'
temporary.write_text(json.dumps(request))
temporary.replace(root / 'request.json')
deadline = time.monotonic() + float(seconds)
while time.monotonic() < deadline:
    try:
        receipt = json.loads((root / 'receipt.json').read_text())
    except FileNotFoundError:
        receipt = {}
    if all(receipt.get(key) == value for key, value in request.items()):
        print(json.dumps(receipt), flush=True)
        break
    time.sleep(0.1)
else:
    pending = root / 'request.json'
    if pending.exists() and json.loads(pending.read_text()).get('request_id') == request['request_id']:
        pending.unlink()
    print(json.dumps({**request, 'status': 'not-observed'}), flush=True)
"""


def inject_gpu_failure(backend, workers, fault, job_id):
    """Ask one allowlisted GPU worker to raise a one-shot error and require its consumption receipt."""
    target = str(fault.get("target", ""))
    if target not in workers:
        raise BenchmarkError(f"GPU faults are not allowed for {target}")
    mode = fault.get("mode", "runtime_error")
    if mode not in {"runtime_error", "out_of_memory"}:
        raise BenchmarkError(f"unsupported GPU failure mode: {mode}")
    try:
        timeout = max(1, min(int(fault.get("detection_timeout_seconds", 60)), 120))
    except (ValueError, TypeError) as error:
        raise BenchmarkError("GPU fault timeout must be an integer") from error
    output = backend.kubectl(
        "exec",
        "-n",
        backend.namespace,
        f"deployment/{target}",
        "-c",
        target,
        "--",
        "/opt/cpython/bin/python3",
        "-c",
        GPU_FAULT_SCRIPT,
        job_id,
        mode,
        str(timeout),
        timeout=timeout + 15,
    )
    try:
        receipt = json.loads(output.stdout)
    except (ValueError, TypeError) as error:
        raise BenchmarkError("GPU injection returned an invalid receipt") from error
    if not isinstance(receipt, dict):
        raise BenchmarkError("GPU injection returned an invalid receipt")
    if (
        receipt.get("job_id") != job_id
        or receipt.get("mode") != mode
        or receipt.get("status") not in {"injected", "not-observed"}
    ):
        raise BenchmarkError("GPU injection receipt does not match the requested fault")
    if receipt["status"] == "injected" and receipt.get("device") not in {"cuda", "mps"}:
        raise BenchmarkError("GPU injection did not confirm an accelerator")
    return {**receipt, "injection_level": "application-boundary", "hardware_failure": False}


def main():
    """Inject a cooperative error into a native local worker and report its receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--mode", choices=("runtime_error", "out_of_memory"), default="runtime_error")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    timeout = max(1, min(args.timeout, 120))
    result = subprocess.run(
        [sys.executable, "-c", GPU_FAULT_SCRIPT, args.job_id, args.mode, str(timeout)],
        capture_output=True,
        text=True,
        timeout=timeout + 15,
        check=False,
    )
    if result.returncode:
        print(result.stderr, file=sys.stderr)
        return result.returncode
    print(result.stdout, end="")
    return 0 if json.loads(result.stdout).get("status") == "injected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
