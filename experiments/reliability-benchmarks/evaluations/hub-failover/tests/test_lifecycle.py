"""Failure lifecycle, immutable inputs and controller evidence contracts."""

import asyncio
import json
import os
import signal
import sys
import tempfile
import threading
import time
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import study
import topology
import workload
from topology import Deployment, save

TARGET = {"host": "unix:///test/docker.sock", "id": "daemon"}


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_summary_is_refreshed_before_a_later_repetition_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)

            async def attempt(settings, _snapshot, path, **_kwargs):
                if path.parent.name == "r01":
                    raise RuntimeError("later failure")
                report = {
                    "settings": settings,
                    "snapshot": {"image_id": "core"},
                    "status": "passed",
                    "preflight": {"images": {}},
                    "fleet_recovery_seconds": 2,
                    "timing": {
                        "recovery_seconds": 1,
                        "detection_seconds": 1,
                        "post_down_seconds": 1,
                    },
                }
                save(path / "report.json", report)
                return report

            with (
                patch.object(
                    study,
                    "preflight",
                    return_value={"images": {}, "docker_target": TARGET},
                ),
                patch.object(study, "attempt", side_effect=attempt),
                self.assertRaisesRegex(RuntimeError, "later failure"),
            ):
                await study.suite(
                    {"fleet_size": 10, "warmup_seconds": 0, "repetitions": 1},
                    {},
                    output,
                )
            self.assertIn("smoke,10,", (output / "summary.csv").read_text())

    def test_inventory_requires_the_same_daemon_after_all_three_reads(self):
        with (
            patch("topology.command", side_effect=["daemon", "", "", "", "replacement"]),
            self.assertRaisesRegex(RuntimeError, "identity changed"),
        ):
            topology.verify_absent("et-hub-study-0123456789ab", TARGET)

    def test_failed_teardown_keeps_private_compose_until_reconciled(self):
        settings = json.loads((study.ROOT / "settings.json").read_text())
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            project = Deployment(settings, "image", output, docker_target=TARGET)
            with (
                patch.object(topology, "verify_daemon"),
                patch.object(topology, "verify_absent", return_value=["Owned volumes remain"]),
                patch.object(project, "_logs"),
                patch.object(project, "compose", side_effect=TimeoutError("down timed out")),
            ):
                self.assertEqual(project.cleanup(), ["Owned volumes remain"])
            private = project.directory / "compose.json"
            self.assertTrue(private.is_file())
            save(
                output / "report.json",
                {
                    "cleanup_verified": False,
                    "docker_target": TARGET,
                    "project": project.name,
                },
            )
            with patch.object(study, "verify_absent", return_value=[]):
                self.assertTrue(study.reconcile(output / "report.json"))
            self.assertFalse(private.exists())

    async def test_local_http_requests_bypass_proxy_environment_and_preserve_put(self):
        requests = []

        async def handler(reader, writer):
            headers = await reader.readuntil(b"\r\n\r\n")
            size = next(
                (
                    int(line.split(b":", 1)[1])
                    for line in headers.splitlines()
                    if line.lower().startswith(b"content-length:")
                ),
                0,
            )
            requests.append((headers.split()[0], await reader.readexactly(size)))
            writer.write(b"HTTP/1.0 200 OK\r\nContent-Length: 2\r\nX-Benchmark-Hub: b\r\n\r\nok")
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        async with server:
            with (
                patch.dict(
                    os.environ,
                    {
                        "HTTP_PROXY": "http://127.0.0.1:1",
                        "http_proxy": "http://127.0.0.1:1",
                        "NO_PROXY": "",
                        "no_proxy": "",
                    },
                ),
                patch("urllib.request._opener", None),
            ):
                url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/health"
                self.assertEqual(await asyncio.to_thread(workload.http, url), (b"ok", "b"))
                self.assertEqual(await asyncio.to_thread(workload.http, url, b"checkpoint"), (b"ok", "b"))
        self.assertEqual(requests, [(b"GET", b""), (b"PUT", b"checkpoint")])

    def test_large_explicit_fleets_reuse_checked_ports_without_overriding_operator_values(self):
        self.assertEqual(study.checked_ports({"fleet_size": 1000}), {"fleet_size": 1000})
        settings = study.checked_ports({"fleet_size": 25000})
        self.assertEqual(settings["source_port_base"], 1024)
        self.assertEqual(settings["source_port_limit"], 32768)
        custom = {"fleet_size": 25000, "source_port_base": 22000, "source_port_limit": 65536}
        self.assertEqual(study.checked_ports(custom), custom)
        self.assertEqual(study.large_fleet(custom)["source_port_base"], 22000)

    async def test_standalone_smoke_is_admitted_at_ten_agents_but_run_at_full_size(self):
        for smoke in (True, False):
            with (
                self.subTest(smoke=smoke),
                tempfile.TemporaryDirectory() as folder,
                patch.object(study, "preflight", return_value={"images": {}, "docker_target": TARGET}) as check,
                patch.object(study, "attempt", side_effect=RuntimeError("admitted")),
            ):
                with self.assertRaisesRegex(RuntimeError, "admitted"):
                    await study.suite(
                        {"fleet_size": 25000, "warmup_seconds": 30, "repetitions": 1}, {}, Path(folder), smoke=smoke
                    )
                self.assertEqual(check.call_args.args[0]["fleet_size"], 10 if smoke else 25000)

    def test_pre_start_attempt_can_resume_without_a_report_or_docker(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            save(output / "startup.json", {"phase": "pre-start"})
            with patch.object(study, "verify_absent") as inventory:
                self.assertTrue(study.reconcile(output / "report.json"))
                inventory.assert_not_called()
            save(output / "startup.json", {"phase": "starting"})
            self.assertFalse(study.reconcile(output / "report.json"))

    async def test_attempt_is_not_published_when_its_pre_start_journal_cannot_be_written(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(study, "save", side_effect=OSError("journal failed")),
        ):
            output = Path(folder) / "attempt-01"
            with self.assertRaisesRegex(OSError, "journal failed"):
                await study.attempt({}, {}, output)
            self.assertFalse(output.exists())

    async def recovery_fixture(self, *, ack_error=False, registration_seconds=5, closed_before_ack=False):
        settings = json.loads((study.ROOT / "settings.json").read_text())
        settings.update(
            fleet_size=1,
            warmup_seconds=0,
            recovery_seconds=5,
            registration_seconds=registration_seconds,
            probe_seconds=0.01,
        )
        project = SimpleNamespace(name="owned", paused=False, stats=lambda: "")
        project.pause = lambda: setattr(project, "paused", True)
        events, payload, connections = [], [], []
        closed_once = False
        report = {"health_samples": [{"at": time.monotonic(), "status": "UP", "check_status": "L7OK"}]}
        report["request_starts"] = []
        pending = {"type": "et-agent-message", "message_id": "message", "message": {"nonce": project.name}}
        fleet = Mock(failures=0)
        fleet.close = AsyncMock()
        fleet.sample.side_effect = lambda: {
            "at": time.monotonic(),
            "connected": 1,
            "recent_heartbeats": 1,
            "on_a": int(not project.paused),
            "on_b": int(project.paused),
        }

        async def registered(*_args, **_kwargs):
            acknowledgement = {
                "type": "et-message-status",
                "status": "acknowledged",
                "message_id": "message",
            }
            messages = (
                [
                    {
                        "type": "et-message-status",
                        "status": "delivered",
                        "message_id": "message",
                    },
                    acknowledgement,
                ]
                if not connections
                else [pending, acknowledgement]
                if len(connections) >= 2
                else [pending]
            )
            socket = SimpleNamespace(
                response=SimpleNamespace(headers={"X-Benchmark-Hub": "a" if len(connections) == 1 else "b"}),
                close=AsyncMock(),
                state=workload.State.OPEN,
                messages=deque(messages),
            )
            connections.append(socket)
            return socket

        async def send(_socket, **message):
            if message["type"] == "et-message-ack":
                events.append("ack")

        async def until(_socket, predicate):
            value = _socket.messages.popleft()
            self.assertTrue(
                predicate(value),
                "Acknowledgement was awaited on the wrong socket or consumed twice",
            )
            if ack_error and value.get("status") == "acknowledged":
                raise TimeoutError("acknowledgement observation timed out")
            if value.get("status") == "acknowledged":
                events.append("sender-confirmed" if _socket is connections[0] else "recipient-confirmed")
            return value

        def http(url, data=None):
            nonlocal closed_once
            if url.endswith("/info"):
                return b"{}", None
            if url.endswith("/stats;csv"):
                report["request_starts"].append(time.monotonic())
                return b"# pxname,svname,status,check_status\nhubs,a,UP,L7OK\n", None
            if data is not None:
                payload.append(data)
                return b"", None
            if project.paused:
                if not events and not ack_error:
                    events.append("read-failed")
                    raise TimeoutError("transient artifact read failure")
                events.append("read-ok")
                if closed_before_ack and not closed_once:
                    connections[-1].state = workload.State.CLOSED
                    closed_once = True
            return payload[0], "b"

        with (
            patch.object(workload, "Fleet", return_value=fleet),
            patch.object(workload, "registered", side_effect=registered),
            patch.object(workload, "send", side_effect=send),
            patch.object(workload, "until", side_effect=until),
            patch.object(workload, "http", side_effect=http),
            patch.object(
                workload, "summarize", return_value={"unhealthy_observed_at": time.monotonic(), "recovery_seconds": 1}
            ),
        ):
            if registration_seconds == 0:
                with self.assertRaisesRegex(RuntimeError, "registration deadline"):
                    await workload.exercise(settings, project, report)
                self.assertFalse(project.paused)
            elif ack_error:
                with self.assertRaisesRegex(RuntimeError, "failed after acknowledgement"):
                    await workload.exercise(settings, project, report)
            else:
                await workload.exercise(settings, project, report)
        return events, report

    async def test_artifact_retry_keeps_the_delivery_unacknowledged_until_integrity_is_verified(
        self,
    ):
        events, report = await self.recovery_fixture()
        self.assertEqual(
            events,
            [
                "read-failed",
                "read-ok",
                "ack",
                "recipient-confirmed",
                "sender-confirmed",
            ],
        )
        self.assertTrue(report["proof"]["artifact_intact"])
        self.assertGreaterEqual(report["health_samples"][1]["at"], report["request_starts"][0])
        self.assertLessEqual(report["health_samples"][1]["requested"], report["request_starts"][0])

    async def test_complete_registration_sample_after_deadline_does_not_trigger_a_fault(self):
        events, report = await self.recovery_fixture(registration_seconds=0)
        self.assertEqual(events, [])
        self.assertNotIn("fault_at", report)

    async def test_closed_replacement_retries_before_sending_the_acknowledgement(self):
        events, report = await self.recovery_fixture(closed_before_ack=True)
        self.assertEqual(events.count("read-ok"), 2)
        self.assertEqual(events.count("ack"), 1)
        self.assertTrue(report["proof"]["sender_ack"])

    async def test_acknowledgement_observer_failure_does_not_retry_an_acknowledged_delivery(self):
        events, report = await self.recovery_fixture(ack_error=True)
        self.assertEqual(events, ["read-ok", "ack"])
        self.assertNotIn("proof", report)

    async def test_sigterm_during_cleanup_waits_for_report_and_ignores_repeated_signal(self):
        script = """
import asyncio, pathlib, sys, time
from unittest.mock import patch
import study
root=pathlib.Path(sys.argv[1])
class Project:
    name='et-hub-study-0123456789ab'
    collection_errors=[]
    cleanup_diagnostics=[]
    pending_pause=None
    def __init__(self,*args,**kwargs): pass
    def compose(self,*args): pass
    def cleanup(self):
        (root/'cleaning').touch()
        deadline=time.monotonic()+5
        while not (root/'release').exists():
            if time.monotonic()>deadline: raise RuntimeError('cleanup fixture timed out')
            time.sleep(.01)
        (root/'removed').touch()
        return []
async def operation(*args): pass
async def suite(*args,**kwargs):
    return await study.attempt({'port_base':1},{'image_id':'core'},root/'attempt',operation=operation)
with patch.object(study,'preflight',return_value={'docker_target':{'host':'unix:///test','id':'daemon'}}), \
    patch.object(study,'Deployment',Project),patch.object(study,'http'),patch.object(study,'suite',suite):
    try: asyncio.run(study.run_interruptible({}, {}, root))
    except asyncio.CancelledError: pass
"""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-c", script, folder, cwd=study.ROOT, stderr=asyncio.subprocess.PIPE
            )
            try:
                async with asyncio.timeout(5):
                    while not (root / "cleaning").exists():
                        if process.returncode is not None:
                            self.fail((await process.communicate())[1].decode())
                        await asyncio.sleep(0.01)
                os.kill(process.pid, signal.SIGTERM)
                await asyncio.sleep(0.05)
                os.kill(process.pid, signal.SIGTERM)
                await asyncio.sleep(0.05)
                self.assertIsNone(process.returncode)
                (root / "release").touch()
                _, errors = await asyncio.wait_for(process.communicate(), timeout=5)
                self.assertEqual(process.returncode, 0, errors.decode())
                self.assertTrue((root / "removed").exists())
                report = json.loads((root / "attempt/report.json").read_text())
                self.assertTrue(report["cleanup_verified"])
                self.assertEqual(report["status"], "not-established")
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()

    def test_teardown_uses_recorded_endpoint_and_refuses_a_changed_daemon(self):
        settings = json.loads((study.ROOT / "settings.json").read_text())
        with tempfile.TemporaryDirectory() as folder:
            project = Deployment(settings, "image", Path(folder), docker_target=TARGET)
            calls = []

            def command(args, **_kwargs):
                calls.append(args)
                return "daemon" if "info" in args else ""

            with (
                patch.dict(os.environ, {"DOCKER_CONTEXT": "another-context"}),
                patch.object(topology, "command", side_effect=command),
            ):
                self.assertEqual(project.cleanup(), [])
            self.assertTrue(all(args[:3] == ["docker", "--host", TARGET["host"]] for args in calls))
            with patch.object(topology, "command", return_value="replaced") as command:
                self.assertIn("identity changed", project.cleanup()[0])
                self.assertEqual(command.call_count, 1)

    def test_context_resolution_respects_environment_precedence(self):
        values = [json.dumps([{"Endpoints": {"docker": {"Host": TARGET["host"]}}}]), json.dumps({"ID": "daemon"})]
        with (
            patch.dict(os.environ, {"DOCKER_CONTEXT": "chosen", "DOCKER_HOST": "unix:///ignored"}),
            patch.object(topology, "command", side_effect=values) as command,
        ):
            self.assertEqual(topology.resolve_docker(), TARGET)
            self.assertEqual(command.call_args_list[0].args[0], ["docker", "context", "inspect", "chosen"])

    async def test_shared_output_blocks_concurrent_suite_before_reconciliation(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def run(*_args, **_kwargs):
            entered.set()
            await release.wait()

        with tempfile.TemporaryDirectory() as folder, patch.object(study, "run_suite", side_effect=run) as run:
            output = Path(folder)
            first = asyncio.create_task(study.suite({}, {}, output))
            try:
                await asyncio.wait_for(entered.wait(), timeout=1)
                with self.assertRaises(BlockingIOError):
                    await study.suite({}, {}, output / ".." / output.name)
                self.assertEqual(run.call_count, 1)
            finally:
                release.set()
                await first

    async def test_preflight_failure_requires_no_teardown_and_allows_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "smoke/attempt-01"
            with (
                patch.object(study, "preflight", side_effect=RuntimeError("port occupied")),
                patch.object(study, "Deployment") as deployment,
            ):
                report = await study.attempt({}, {}, output)
            self.assertTrue(report["cleanup_verified"])
            self.assertIn("port occupied", report["error"])
            deployment.assert_not_called()
            with (
                patch.object(study, "preflight", return_value={"images": {}, "docker_target": TARGET}),
                patch.object(study, "attempt", side_effect=RuntimeError("retry reached")),
                self.assertRaisesRegex(RuntimeError, "retry reached"),
            ):
                await study.suite({"warmup_seconds": 0, "repetitions": 1}, {}, Path(folder), smoke=True)

    async def test_cancellation_waits_for_inflight_startup_before_teardown(self):
        started, release = threading.Event(), threading.Event()
        project = Mock(name="deployment")
        project.name = "owned"
        project.collection_errors = []
        project.cleanup_diagnostics = []
        project.cleanup.return_value = []

        def compose(*_args):
            started.set()
            if not release.wait(3):
                raise RuntimeError("test startup was not released")

        project.compose.side_effect = compose
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(study, "preflight", return_value={"docker_target": TARGET}),
            patch.object(study, "Deployment", return_value=project),
        ):
            task = asyncio.create_task(study.attempt({}, {"image_id": "core"}, Path(folder) / "attempt"))
            deadline = asyncio.get_running_loop().time() + 2
            try:
                while not started.is_set():
                    if task.done():
                        self.fail("Attempt exited before startup: " + str(await task))
                    if asyncio.get_running_loop().time() >= deadline:
                        self.fail("Startup signal did not arrive within two seconds")
                    await asyncio.sleep(0.001)
            except BaseException:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
                raise
            task.cancel()
            await asyncio.sleep(0.01)
            project.cleanup.assert_not_called()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            project.cleanup.assert_called_once()

    async def test_suite_pins_supporting_images_and_rejects_mismatched_evidence(self):
        metadata = {"core": {"id": "sha256:core"}, "proxy:tag": {"id": "sha256:proxy"}}
        for mismatch in (False, True):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as folder:
                output = Path(folder)

                async def attempt(settings, _snapshot, path, *, expected_images, docker_target, mismatch=mismatch):
                    self.assertEqual(docker_target, TARGET)
                    self.assertEqual(settings["gateway_image"], "sha256:proxy")
                    self.assertEqual(expected_images, {"sha256:core", "sha256:proxy"})
                    images = metadata if not mismatch else {"proxy": {"id": "sha256:changed"}}
                    report = {"status": "passed", "preflight": {"images": images}, "timing": {"recovery_seconds": 1}}
                    save(path / "report.json", report)
                    return report

                with (
                    patch.object(study, "preflight", return_value={"images": metadata, "docker_target": TARGET}),
                    patch.object(study, "attempt", side_effect=attempt),
                    patch.object(study, "export_csv"),
                ):
                    operation = study.suite(
                        {"gateway_image": "proxy:tag", "warmup_seconds": 0, "repetitions": 1}, {}, output, smoke=True
                    )
                    if mismatch:
                        with self.assertRaisesRegex(RuntimeError, "receipt not written"):
                            await operation
                        self.assertFalse((output / "smoke/receipt.json").exists())
                    else:
                        await operation
                        self.assertTrue((output / "smoke/receipt.json").is_file())

    async def test_cancelled_command_failure_does_not_invalidate_verified_teardown(self):
        for stage in ("startup", "pause"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as folder:
                started, release = threading.Event(), threading.Event()
                project = Mock()
                project.name = "owned"
                project.collection_errors, project.cleanup_diagnostics, project.pending_pause = [], [], None
                project.cleanup.return_value = []

                def blocked(*_args, started=started, release=release):
                    started.set()
                    if not release.wait(2):
                        raise RuntimeError("test command was not released")
                    raise RuntimeError("command failed after cancellation")

                project.compose.side_effect = blocked if stage == "startup" else None

                async def operation(_settings, project, _report):
                    project.pending_pause = asyncio.create_task(asyncio.to_thread(blocked))
                    await asyncio.shield(project.pending_pause)

                with (
                    patch.object(study, "preflight", return_value={"docker_target": TARGET}),
                    patch.object(study, "Deployment", return_value=project),
                    patch.object(study, "http"),
                ):
                    output = Path(folder) / "attempt"
                    task = asyncio.create_task(
                        study.attempt({"port_base": 1}, {"image_id": "core"}, output, operation=operation)
                    )
                    try:
                        async with asyncio.timeout(1):
                            while not started.is_set():
                                if task.done():
                                    self.fail("Attempt exited before command: " + str(await task))
                                await asyncio.sleep(0.001)
                        task.cancel()
                        await asyncio.sleep(0.01)
                        project.cleanup.assert_not_called()
                    finally:
                        release.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    report = json.loads((output / "report.json").read_text())
                    self.assertTrue(report["cleanup_verified"])
                    self.assertIn("command failed after cancellation", report["command_errors"][0])

    def test_log_collection_error_does_not_mean_resources_remain(self):
        settings = json.loads((study.ROOT / "settings.json").read_text())
        with tempfile.TemporaryDirectory() as folder:
            project = Deployment(settings, "image", Path(folder), docker_target=TARGET)
            with (
                patch.object(project, "_logs", side_effect=TimeoutError("logs timed out")),
                patch.object(project, "compose"),
                patch(
                    "topology.command",
                    side_effect=["daemon", "daemon", "", "", "", "daemon"],
                ),
            ):
                self.assertEqual(project.cleanup(), [])
            self.assertEqual(project.collection_errors, ["logs timed out"])

    def test_manual_cleanup_checks_daemon_and_all_inventories(self):
        for remaining in ("", "container"):
            with (
                self.subTest(remaining=remaining),
                tempfile.TemporaryDirectory() as folder,
            ):
                report = Path(folder) / "report.json"
                save(
                    report, {"cleanup_verified": False, "docker_target": TARGET, "project": "et-hub-study-0123456789ab"}
                )
                with (
                    patch(
                        "topology.command",
                        side_effect=["daemon", remaining, "", "", "daemon"],
                    ) as inventory,
                ):
                    self.assertEqual(study.reconcile(report), not remaining)
                    self.assertEqual(inventory.call_count, 5)

    def test_large_controller_fleet_reuses_checked_port_policy(self):
        settings = study.large_fleet({"fleet_size": 3})
        settings.update(fleet_size=25000, repetitions=1)
        self.assertEqual(settings["source_port_base"], 1024)
        self.assertEqual(settings["source_port_limit"], 32768)
        self.assertEqual(settings["fleet_size"], 25000)

    def test_controller_receives_json_recovery_evidence_without_smoke_incidents(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            save(output / "inputs.json", {"signature": "same"})
            for name in ("smoke", "r01"):
                report = output / name / "attempt-01/report.json"
                save(
                    report,
                    {
                        "status": "passed",
                        "timing": {"unhealthy_observed_at": 10, "down_observed_at": 11, "work_restored_at": 12},
                    },
                )
                save(
                    report.parent.parent / "receipt.json",
                    {"signature": "same", "report": "attempt-01/report.json", "sha256": study.digest(report)},
                )
            result = study.adapter_result(output)
            evidence = json.loads(Path(result["report"]).read_text())
            incidents = evidence["recovery"]["incidents"]
            self.assertEqual(len(incidents), 1)
            self.assertEqual(incidents[0]["recovered_at"] - incidents[0]["observed_failure_at"], 2)
