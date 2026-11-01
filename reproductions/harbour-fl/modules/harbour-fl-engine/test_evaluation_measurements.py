"""Measure transfer failures and final-model robustness without mutating training conditions."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from harbour_fl import data, quality
from harbour_fl.learning import _import_torch
from harbour_fl.runtime import VesselRuntime
from harbour_fl.transfers import measure_transfer
from test_fixtures import prepared


class TransferInstrumentationTests(unittest.TestCase):
    def test_success_records_bytes_duration_and_one_correlated_attempt(self):
        runtime = SimpleNamespace(_event=Mock())
        with patch("harbour_fl.transfers.time.monotonic", side_effect=[1, 1.5]):
            value, seconds = measure_transfer(runtime, "download", 0, lambda: b"abc")
        self.assertEqual((value, seconds), (b"abc", 0.5))
        start, end = runtime._event.call_args_list
        self.assertEqual(start.kwargs["transfer_id"], end.kwargs["transfer_id"])
        self.assertEqual(end.kwargs["payload_bytes"], 3)
        self.assertEqual(end.kwargs["status"], "success")

    def test_missing_payload_and_storage_timeout_are_unsuccessful(self):
        """Verify missing payload and storage timeout are unsuccessful."""
        for operation, expected in ((lambda: None, "failure"), (Mock(side_effect=TimeoutError()), "timeout")):
            runtime = SimpleNamespace(_event=Mock())
            with self.assertRaises((RuntimeError, TimeoutError)):
                measure_transfer(runtime, "download", 0, operation)
            self.assertEqual(runtime._event.call_args.kwargs["status"], expected)


class PredictionQualityTests(unittest.TestCase):
    def test_augmented_coordinator_reports_an_explicit_clean_reference_under_an_input_fault(self):
        """Verify augmented coordinator reports an explicit clean reference under an input fault."""
        engine = VesselRuntime()
        engine.state.update(role="coordinator", agent_id="coordinator", storage=Mock(), send=Mock(), events=Mock())
        engine.state["storage"].get.return_value = None
        engine.state["events"].checkpoints.return_value.load.return_value = None
        engine._collect_round = Mock(
            return_value=[
                {
                    "client_id": "vessel_1",
                    "metrics": {"validation_accuracy": 0.5, "train_loss": 1.0, "validation_loss": 1.0},
                }
            ]
        )

        def evaluate(_model, _rows, _labels, _torch, *, input_condition=None):
            """Return distinct clean and fault-condition metrics to expose a mislabeled clean reference."""
            return {
                "accuracy": 0.8 if input_condition == {} else 0.3,
                "loss": 1.0,
                "macro_f1": 0.6,
                "confusion_matrix": [[1, 0], [0, 1]],
            }

        with (
            patch.dict(
                os.environ,
                {
                    "HARBOUR_FL_AUTO_START": "true",
                    "HARBOUR_FL_START_DELAY_SECONDS": "0",
                    "HARBOUR_FL_ROUNDS": "1",
                    "HARBOUR_FL_ADAPTATION": "true",
                    "HARBOUR_FL_EVALUATE_DEGRADED_INPUTS": "true",
                    "HARBOUR_FL_IMAGE_NOISE_STD": "0.2",
                    "HARBOUR_FL_READINESS_FILE": "",
                },
            ),
            patch("harbour_fl.coordinator._data_root", return_value=Path("unused")),
            patch("harbour_fl.coordinator._labels", return_value=["a", "b"]),
            patch("harbour_fl.coordinator._load_rows", return_value=[1]),
            patch("harbour_fl.coordinator._fed_avg"),
            patch("harbour_fl.coordinator._evaluate", side_effect=evaluate),
            patch("harbour_fl.quality._evaluate", side_effect=evaluate),
        ):
            engine._coordinator_main()
        engine._collect_round.assert_called_once()
        metrics = {
            call.kwargs["condition"]: call.kwargs["metrics"]
            for call in engine.state["events"].emit.call_args_list
            if call.args[0] == "quality-evaluation"
        }
        self.assertEqual(metrics["clean"]["accuracy"], 0.8)
        self.assertEqual(len(metrics), 8)
        self.assertEqual(engine.state["coordinator_status"]["status"], "run-completed")
        self.assertTrue(engine.state["completed"])

    def test_suite_reuses_clean_reference_and_does_not_train(self):
        """Verify suite reuses clean reference and does not train."""
        model, emit = Mock(), Mock()
        reference = {"accuracy": 0.7, "macro_f1": 0.6}
        with patch("harbour_fl.quality._evaluate", return_value={"accuracy": 0.5, "macro_f1": 0.4}) as evaluate:
            results = quality.evaluate_inputs(model, [1, 2], ["a", "b"], Mock(), reference, emit, 8)
        self.assertEqual(len(results), 8)
        self.assertIs(results["clean"], reference)
        self.assertEqual(evaluate.call_count, 7)
        model.train.assert_not_called()
        self.assertEqual(
            [call.kwargs["input_condition"] for call in evaluate.call_args_list], list(quality.CONDITIONS.values())[1:]
        )
        self.assertTrue(all(call.kwargs["scope"] == "prediction-time" for call in emit.call_args_list))

    def test_explicit_prediction_condition_does_not_inherit_training_faults(self):
        torch = _import_torch()
        with tempfile.TemporaryDirectory() as directory:
            root = prepared(Path(directory))
            rows = data._load_rows(root, root / "test_manifest.csv")
            with patch.dict(os.environ, {"HARBOUR_FL_MISSING_MODALITY": "audio", "HARBOUR_FL_IMAGE_NOISE_STD": "0.2"}):
                clean_audio, clean_image, _targets = data._materialize(rows, ["a", "b"], torch, {})
                masked_audio, kept_image, _targets = data._materialize(
                    rows, ["a", "b"], torch, {"missing_modality": "audio"}
                )
            self.assertGreater(float(clean_audio.sum()), 0)
            self.assertGreater(float(clean_image.sum()), 0)
            self.assertEqual(float(masked_audio.sum()), 0)
            self.assertTrue(torch.equal(kept_image, clean_image))


if __name__ == "__main__":
    unittest.main()
