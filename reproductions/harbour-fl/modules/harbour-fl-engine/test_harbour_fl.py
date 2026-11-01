import json
import os
import queue
import unittest
from unittest import mock

from harbour_fl import checkpoints, learning, settings
from harbour_fl.runtime import VesselRuntime

RUNTIME = VesselRuntime()


class MemoryStorage:
    def __init__(self):
        self.values = {}

    def put(self, name, value):
        self.values[name] = bytes(value)


class ConfusionMatrixArtifactTests(unittest.TestCase):
    def test_writes_html_heatmap_and_json_for_one_round(self):
        """Verify writes html heatmap and json for one round."""
        storage = MemoryStorage()
        RUNTIME.state["storage"] = storage
        RUNTIME.state["agent_id"] = "coordinator"

        artifacts = RUNTIME._confusion_matrix_artifacts(
            "job-1",
            2,
            ["cargo", "passenger & ferry"],
            [[8, 1], [2, 5]],
        )

        self.assertEqual(
            [item["artifact_path"] for item in artifacts],
            [
                "evaluation/round-002/confusion-matrix.html",
                "evaluation/round-002/confusion-matrix.json",
            ],
        )
        html = storage.values["job-1-confusion-round-002.html"].decode()
        self.assertIn("Actual ↓ / Predicted →", html)
        self.assertIn("passenger &amp; ferry", html)
        self.assertIn("rgba(24, 119, 242", html)
        self.assertIn(">8</td>", html)
        json_text = storage.values["job-1-confusion-round-002.json"].decode()
        self.assertIn('"matrix": [', json_text)
        self.assertIn('"round": 2', json_text)

    def test_mlflow_round_metrics_are_loss_centric_per_client(self):
        display_metrics = {
            "global_test_loss": 1.2,
            "global_test_accuracy": 0.7,
            "global_test_macro_f1": 0.5,
            "mean_client_validation_accuracy": 0.6,
            "participating_clients": 2.0,
        }
        updates = [
            {"client_id": "vessel_1", "metrics": {"train_loss": 0.9, "validation_loss": 1.0}},
            {"client_id": "vessel_2", "metrics": {"train_loss": 0.8, "validation_loss": 1.1}},
        ]

        result = RUNTIME._mlflow_round_metrics(display_metrics, updates)

        self.assertEqual(result["global_test_loss"], 1.2)
        self.assertEqual(result["client.vessel_1.training_loss"], 0.9)
        self.assertEqual(result["client.vessel_2.validation_loss"], 1.1)
        self.assertNotIn("global_test_accuracy", result)
        self.assertNotIn("global_test_macro_f1", result)
        self.assertNotIn("mean_client_validation_accuracy", result)

    def test_mlflow_events_use_generic_namespaced_schema(self):
        class Sender:
            def __init__(self):
                self.value = None

            def text(self, value):
                self.value = json.loads(value)

        sender = Sender()
        RUNTIME.state["send"] = sender
        RUNTIME._emit_mlflow("metrics", "job-1:round:0", "job-1", step=0, metrics={"loss": 1.0})

        self.assertEqual(sender.value["schema_version"], 1)
        self.assertEqual(sender.value["producer"], "harbour-fl-demo")
        self.assertEqual(sender.value["run_key"], "job-1")


class ContributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load the training dependency required by the contribution-validation tests."""
        try:
            cls.torch = learning._import_torch()
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_nonfinite_update_is_rejected_and_valid_quorum_survives(self):
        """Verify nonfinite update is rejected and valid quorum survives."""

        class Storage(MemoryStorage):
            def get(self, _bucket, name):
                return self.values.get(name)

        storage = Storage()
        model = self.torch.nn.Linear(2, 2)
        storage.put("global.pt", checkpoints._checkpoint_bytes(model, ["a", "b"], 0, self.torch))
        storage.put("good.pt", checkpoints._checkpoint_bytes(model, ["a", "b"], 0, self.torch))
        for parameter in model.parameters():
            parameter.data.fill_(float("nan"))
        storage.put("bad.pt", checkpoints._checkpoint_bytes(model, ["a", "b"], 0, self.torch))
        updates = queue.Queue()
        for client in ("bad", "good"):
            updates.put(
                {
                    "type": settings.UPDATE_TYPE,
                    "job_id": "job",
                    "round": 0,
                    "status": "ready",
                    "client_id": client,
                    "bucket": client,
                    "filename": f"{client}.pt",
                    "samples": 1,
                }
            )
        events = mock.Mock()
        with (
            mock.patch.dict(
                RUNTIME.state,
                storage=storage,
                agent_id="coordinator",
                update_queue=updates,
                send=mock.Mock(),
                events=events,
            ),
            mock.patch.dict(os.environ, {"HARBOUR_FL_MIN_CLIENTS": "1", "HARBOUR_FL_ROUND_TIMEOUT_SECONDS": "0.05"}),
        ):
            result = RUNTIME._collect_round("job", 0, "global.pt", ("bad", "good"))
        self.assertEqual([update["client_id"] for update in result], ["good"])
        self.assertTrue(any(call.args[0] == "update-rejected" for call in events.emit.call_args_list))

    def test_repeated_round_announcement_replays_cached_update(self):
        sender = mock.Mock()
        response = {"job_id": "job", "type": settings.UPDATE_TYPE, "round": 0}
        with mock.patch.dict(RUNTIME.state, role="client", send=sender, seen_rounds={0}, responses={0: response}):
            RUNTIME.on_text_frame(
                json.dumps({"schema_version": 1, "type": settings.ROUND_TYPE, "job_id": "job", "round": 0})
            )
        self.assertEqual(json.loads(sender.text.call_args.args[0]), response)


if __name__ == "__main__":
    unittest.main()
