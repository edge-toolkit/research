"""Client recovery evidence follows accepted work, not merely completed local computation."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from harbour_fl import coordinator, runtime
from harbour_fl.runtime import VesselRuntime
from harbour_fl.service_checks import TaskChecks
from harbour_fl.settings import COMPLETE_TYPE


def engine_module():
    engine = VesselRuntime()
    engine.state.update(role="client", client_id="vessel_1", events=Mock(), responses={0: {}})
    return engine


class LifecycleTests(unittest.TestCase):
    def test_new_process_clears_stale_readiness_before_announcing_usable_state(self):
        engine = engine_module()
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "ready"
            marker.write_text("old process")
            with patch.dict(
                os.environ,
                {
                    "HARBOUR_FL_READINESS_FILE": str(marker),
                    "HARBOUR_FL_ROLE": "client",
                    "HARBOUR_FL_CLIENT_ID": "vessel_1",
                },
            ):
                engine.init(Mock(), Mock())
                self.assertFalse(marker.exists())
                engine._mark_ready()
                self.assertEqual(marker.read_text(), "ready\n")

    def test_only_matching_uploaded_contribution_acknowledgements_advance_progress(self):
        """Verify only matching uploaded contribution acknowledgements advance progress."""
        engine = engine_module()
        message = {
            "schema_version": 1,
            "type": "harbour-fl-contribution-accepted",
            "job_id": "job",
            "client_id": "vessel_1",
            "round": 0,
        }
        with patch.dict(os.environ, {"HARBOUR_FL_JOB_ID": "job"}):
            for changes in ({"client_id": "vessel_2"}, {"job_id": "other"}, {"round": 1}, {"round": True}):
                engine.on_text_frame(json.dumps({**message, **changes}))
            engine.state["events"].emit.assert_not_called()
            engine.on_text_frame(json.dumps(message))
            engine.state["events"].emit.assert_called_once_with("progress", progress=1)

    def test_completion_from_another_job_is_not_recovery_evidence(self):
        engine = engine_module()
        with patch.dict(os.environ, {"HARBOUR_FL_JOB_ID": "job"}):
            engine.on_text_frame(json.dumps({"schema_version": 1, "type": COMPLETE_TYPE, "job_id": "other"}))
            engine.state["events"].emit.assert_not_called()

    def test_completed_coordinator_reconnect_does_not_start_another_worker(self):
        engine = engine_module()
        engine.state.update(role="coordinator", completed=True, worker=Mock(is_alive=Mock(return_value=False)))
        with patch.object(runtime.threading, "Thread") as thread:
            engine.on_connect("vessel-coordinator")
        thread.assert_not_called()
        engine.state["events"].emit.assert_called_once_with("reconnected")

    def test_failed_worker_can_still_restart_after_reconnection(self):
        engine = engine_module()
        engine.state.update(send=Mock(), worker=Mock(is_alive=Mock(return_value=False)), application_failed=True)
        with patch.object(runtime, "ApplicationRecovery"), patch.object(runtime.threading, "Thread") as thread:
            engine.on_connect("vessel-1")
        self.assertFalse(engine.state["application_failed"])
        thread.return_value.start.assert_called_once()

    def test_worker_initialization_failure_is_reported_without_local_restart(self):
        engine = engine_module()
        with (
            patch.object(engine, "_client_main", side_effect=OSError("dataset unavailable")) as worker,
            self.assertLogs(runtime._logger, level="ERROR"),
        ):
            engine._worker_main()
        worker.assert_called_once()
        engine.state["events"].emit.assert_called_once_with("failed")

    def test_interrupted_coordinator_is_not_ready_before_restoration(self):
        engine = engine_module()
        engine.state.update(storage=Mock(), send=Mock(), agent_id="coordinator")
        engine.state["storage"].get.return_value = b'{"progress": 1}'
        with (
            patch.dict(os.environ, {"HARBOUR_FL_AUTO_START": "true", "HARBOUR_FL_START_DELAY_SECONDS": "20"}),
            patch.object(engine, "_mark_ready") as ready,
            patch.object(coordinator.time, "sleep") as delay,
            patch.object(coordinator, "_import_torch", side_effect=RuntimeError("initialization stopped")),
            self.assertLogs(runtime._logger, level="ERROR"),
        ):
            engine._coordinator_main()
        ready.assert_not_called()
        delay.assert_not_called()

    def test_terminal_failures_withdraw_file_and_functional_readiness(self):
        """Verify terminal failures withdraw file and functional readiness."""
        for role in ("client", "coordinator"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as directory:
                marker = Path(directory) / "ready"
                engine = engine_module()
                engine.state.update(role=role, storage=Mock(), send=Mock(), agent_id="vessel-coordinator")
                engine.state["storage"].get.return_value = None
                with patch.dict(
                    os.environ,
                    {
                        "HARBOUR_FL_READINESS_FILE": str(marker),
                        "HARBOUR_FL_AUTO_START": "true",
                        "HARBOUR_FL_START_DELAY_SECONDS": "0",
                        "HARBOUR_FL_PROBE_OBSERVER_ID": "observer",
                    },
                ):
                    engine._mark_ready()
                    checks = TaskChecks(engine)
                    self.assertTrue(marker.exists())
                    self.assertTrue(checks.ready())
                    target = (
                        "harbour_fl.client._import_torch"
                        if role == "client"
                        else "harbour_fl.coordinator._import_torch"
                    )
                    with (
                        patch(target, side_effect=RuntimeError("initialization failed")),
                        self.assertLogs(runtime._logger, level="ERROR"),
                    ):
                        engine._worker_main()
                    self.assertTrue(engine.state["application_failed"])
                    self.assertFalse(engine.state["application_ready"])
                    self.assertFalse(marker.exists())
                    self.assertFalse(checks.ready())
