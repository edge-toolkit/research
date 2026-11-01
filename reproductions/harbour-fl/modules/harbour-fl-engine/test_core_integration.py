"""Exercise the real installed Core SDK through the application's integration boundary."""

import json
import unittest
from unittest.mock import patch

from harbour_fl.instrumentation import ApplicationRecovery, checkpoint_key


class Storage:
    agent_id = "worker"

    def __init__(self):
        self.values = {}
        self.fail_pointer = False

    def put(self, name, value):
        """Store checkpoint bytes while optionally failing the latest-pointer commit."""
        if self.fail_pointer and name.endswith("latest.json"):
            raise OSError("injected pointer write failure")
        self.values[name] = value

    def get(self, agent_id, name):
        """Read stored bytes only from the test worker namespace."""
        if agent_id != "worker":
            raise ValueError("unexpected storage namespace")
        return self.values.get(name)


class CoreIntegrationTests(unittest.TestCase):
    def test_interrupted_precommit_keeps_previous_checkpoint(self):
        from harbour_fl.runtime import VesselRuntime

        storage = Storage()
        checkpoints = self.events.checkpoints(storage, "worker")
        checkpoints.save(b"first", 1)
        module = VesselRuntime()
        module.state["events"] = self.events
        with (
            patch.dict("os.environ", {"HARBOUR_FL_CHECKPOINT_WRITE_DELAY_SECONDS": "1"}),
            patch("harbour_fl.coordinator.time.sleep", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            module._save_recovery_checkpoint(checkpoints, b"second", 2)
        self.assertEqual(checkpoints.load(), (b"first", 1))

    def setUp(self):
        self.sent = []
        self.events = ApplicationRecovery(self.sent.append, "job", "worker", "supervisor", "/modules/example")

    def test_real_sdk_sends_direct_ordered_cloudevents_not_research_logs(self):
        """Verify real sdk sends direct ordered cloudevents not research logs."""
        self.events.emit("coordinator-status", status="run-started")
        self.events.emit("operation-started", progress=0)
        self.events.emit("progress", progress=1)
        frames = [json.loads(value) for value in self.sent]
        self.assertEqual([frame["type"] for frame in frames], ["et-send-agent-message"] * 2)
        self.assertEqual([frame["to_agent_id"] for frame in frames], ["supervisor"] * 2)
        self.assertEqual(
            [frame["message"]["type"] for frame in frames], ["et.recovery.started", "et.recovery.progress"]
        )
        self.assertEqual([frame["message"]["data"]["sequence"] for frame in frames], [1, 2])
        self.assertEqual(frames[-1]["message"]["data"]["job_id"], "job")
        self.assertEqual(frames[-1]["message"]["data"]["progress"], 1)

    def test_real_core_checkpoint_preserves_prior_commit_and_rejects_corruption(self):
        storage = Storage()
        checkpoints = self.events.checkpoints(storage, "worker")
        checkpoints.save(b"first", 1)
        storage.fail_pointer = True
        with self.assertRaisesRegex(OSError, "pointer write"):
            checkpoints.save(b"second", 2)
        self.assertEqual(checkpoints.load(), (b"first", 1))
        pointer = json.loads(storage.values[f"{checkpoint_key('job')}-latest.json"])
        storage.values[pointer["filename"]] = b"corrupt"
        with self.assertRaisesRegex(ValueError, "corrupt"):
            checkpoints.load()
