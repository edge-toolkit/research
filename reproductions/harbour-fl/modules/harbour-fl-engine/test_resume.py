"""Exercise a real coordinator process crash and compare resumed weights to uninterrupted execution."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def worker(directory: Path, crash: bool, interval=1):
    """Run a real coordinator process with deterministic updates and an optional abrupt checkpoint-boundary crash."""
    import torch
    from harbour_fl import coordinator
    from harbour_fl.instrumentation import ApplicationRecovery
    from harbour_fl.runtime import VesselRuntime

    os.environ["HARBOUR_FL_MODEL_PROFILE"] = "pretrained"
    module = VesselRuntime()

    class Storage:
        agent_id = "coordinator"

        def put(self, name, value):
            (directory / name).write_bytes(value)

        def get(self, _bucket, name):
            """Return saved checkpoint bytes when the requested recovery artifact exists."""
            path = directory / name
            return path.read_bytes() if path.exists() else None

    class Sender:
        def text(self, _value):
            pass

    module.state.update(
        storage=Storage(),
        send=Sender(),
        agent_id="coordinator",
        events=ApplicationRecovery(Sender().text, "test", "coordinator"),
    )
    coordinator._data_root = lambda: directory
    coordinator._labels = lambda _root: ["a", "b"]
    coordinator._load_rows = lambda *_args: []
    coordinator._evaluate = lambda *_args: {
        "accuracy": 0.5,
        "loss": 1.0,
        "macro_f1": 0.5,
        "confusion_matrix": [[1, 0], [1, 0]],
    }

    def collect(_job, round_number, _filename, _clients):
        """Produce a deterministic synthetic head update or terminate at the selected interrupted-round boundary."""
        if crash and round_number == (4 if interval == 3 else 1):
            os._exit(7)
        checkpoint = coordinator._load_checkpoint(
            (directory / f"test-global-round-{round_number:03d}.pt").read_bytes(), torch
        )
        checkpoint["state_dict"] = {
            key: value + torch.rand_like(value) if value.is_floating_point() else value.clone()
            for key, value in checkpoint["state_dict"].items()
        }
        return [
            {
                "client_id": "client",
                "samples": 1,
                "checkpoint": checkpoint,
                "metrics": {"validation_accuracy": 0.5, "train_loss": 1.0, "validation_loss": 1.0},
            }
        ]

    module._collect_round = collect
    os.environ.update(
        HARBOUR_FL_AUTO_START="true",
        HARBOUR_FL_START_DELAY_SECONDS="0",
        HARBOUR_FL_JOB_ID="test",
        HARBOUR_FL_ROUNDS="6" if interval == 3 else "3",
        HARBOUR_FL_CHECKPOINT_INTERVAL=str(interval),
        HARBOUR_FL_CLIENTS="client",
    )
    module._coordinator_main()
    if not (directory / "test-summary.json").exists():
        raise SystemExit("coordinator did not finish")


class ResumeTests(unittest.TestCase):
    def test_crash_resume_matches_uninterrupted_global_model(self):
        self.check_resume(1)

    def test_stale_checkpoint_repeats_work_and_matches_uninterrupted_model(self):
        self.check_resume(3)

    def check_resume(self, interval):
        """Compare uninterrupted final weights with a coordinator crashed and resumed at the selected cadence."""
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uninterrupted, resumed = root / "uninterrupted", root / "resumed"
            uninterrupted.mkdir()
            resumed.mkdir()
            command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
            settings = ["--interval", str(interval)]
            subprocess.run([*command, str(uninterrupted), *settings], check=True, capture_output=True, timeout=30)
            crashed = subprocess.run(
                [*command, str(resumed), *settings, "--crash"], capture_output=True, timeout=30, check=False
            )
            self.assertEqual(crashed.returncode, 7, crashed.stderr.decode())
            pointer_path = next(resumed.glob("*-latest.json"), None)
            self.assertIsNotNone(pointer_path)
            self.assertEqual(json.loads(pointer_path.read_text())["progress"], 3 if interval == 3 else 1)
            subprocess.run([*command, str(resumed), *settings], check=True, capture_output=True, timeout=30)
            first = torch.load(uninterrupted / "test-final-model.pt", weights_only=True)
            second = torch.load(resumed / "test-final-model.pt", weights_only=True)
            for name in first["state_dict"]:
                self.assertTrue(torch.equal(first["state_dict"][name], second["state_dict"][name]), name)
            self.assertEqual(
                json.loads((resumed / "test-summary.json").read_text())["rounds"], 6 if interval == 3 else 3
            )
            if interval == 1:
                (resumed / "test-summary.json").unlink()
                subprocess.run([*command, str(resumed), *settings], check=True, capture_output=True, timeout=30)
                self.assertTrue((resumed / "test-summary.json").is_file())


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker(
            Path(sys.argv[2]),
            "--crash" in sys.argv,
            int(sys.argv[sys.argv.index("--interval") + 1]),
        )
    else:
        unittest.main()
