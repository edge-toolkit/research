"""Executable adapter template; replace the work loop with calls to the module being tested."""

import json
import os
from pathlib import Path


def main():
    """Read a controller request and write synthetic completion evidence using its output-file contract."""
    request = json.loads(Path(os.environ["BENCHMARK_REQUEST_FILE"]).read_text())
    total = request["parameters"]["work_units"]
    for unit in range(1, total + 1):
        print(f"Module work completed: {unit}/{total}", flush=True)
    result_file = Path(os.environ["BENCHMARK_RESULT_FILE"])
    output = Path(os.environ.get("BENCHMARK_OUTPUT_DIR", str(result_file.parent)))
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "module-result.json"
    artifact.write_text(json.dumps({"completed_work_units": total}) + "\n")
    result_file.write_text(json.dumps({"schema_version": 1, "status": "passed", "report": str(artifact)}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
