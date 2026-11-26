"""Async registered-agent fleet following Core's five-second WebSocket heartbeat workload."""

import asyncio
import csv
import hashlib
import io
import json
import socket as socket_module
import struct
import time
import urllib.request
from itertools import pairwise
from urllib.parse import urlsplit

from websockets.asyncio.client import connect
from websockets.protocol import State

HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http(url, data=None):
    """Perform a bounded storage/provenance request and retain the actual serving backend header."""
    request = urllib.request.Request(url, data=data, method="PUT" if data is not None else "GET")
    with HTTP.open(request, timeout=5) as response:
        return response.read(), response.headers.get("X-Benchmark-Hub")


async def registered(url, identity, *, keepalive=True, source_port=None, reset_on_close=False):
    """Register a stable identity and validate the assigned identity before using the connection."""
    options = {"local_addr": ("127.0.0.1", source_port)} if source_port is not None else {}
    raw = None
    try:
        async with asyncio.timeout(5):
            if source_port is not None and reset_on_close:
                target = urlsplit(url)
                raw = socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM)
                raw.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_REUSEADDR, 1)
                raw.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_LINGER, struct.pack("ii", 1, 0))
                raw.bind(("127.0.0.1", source_port))
                raw.setblocking(False)
                await asyncio.get_running_loop().sock_connect(raw, (target.hostname, target.port or 80))
                options = {"sock": raw}
            socket = await connect(
                url, open_timeout=5, close_timeout=1, ping_interval=5 if keepalive else None, proxy=None, **options
            )
    except BaseException:
        if raw:
            raw.close()
        raise
    try:
        if reset_on_close:
            transport_socket = socket.transport.get_extra_info("socket")
            transport_socket.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_LINGER, struct.pack("ii", 1, 0))
        await socket.send(json.dumps({"type": "et-connect", "agent_id": identity}))
        async with asyncio.timeout(5):
            while True:
                reply = json.loads(await socket.recv())
                if reply.get("type") == "et-connect-ack":
                    if reply.get("agent_id") != identity:
                        raise RuntimeError("Hub did not retain the requested agent identity")
                    return socket
    except BaseException:
        await socket.close()
        raise


async def until(socket, predicate, timeout=5):
    """Read protocol messages until the requested evidence arrives, with one overall deadline."""
    async with asyncio.timeout(timeout):
        while True:
            message = json.loads(await socket.recv())
            if message.get("type") == "et-invalid":
                raise RuntimeError("Hub rejected protocol operation: " + str(message))
            if predicate(message):
                return message


async def send(socket, **message):
    """Send an explicitly typed Core protocol message."""
    await socket.send(json.dumps(message))


class Fleet:
    """Keep all agents reading and heartbeating; reconnect only through the stable gateway endpoint."""

    def __init__(self, settings):
        self.settings = settings
        self.url = f"ws://127.0.0.1:{settings['port_base']}/ws"
        self.connections = {}
        self.beats = {}
        self.backends = {}
        self.failures = 0
        self.connection_errors = {}
        self.semaphore = asyncio.Semaphore(settings.get("registration_concurrency", 100))
        self.tasks = []

    def start(self):
        """Start a bounded registration ramp rather than a simultaneous connection storm."""
        self.tasks = [asyncio.create_task(self.agent(i)) for i in range(self.settings["fleet_size"])]

    async def agent(self, index):
        """Keep the same identity across a disconnect and acknowledge periodic WebSocket pings."""
        identity = f"{self.settings.get('identity_prefix', 'fleet')}-{index:05}"
        while True:
            socket = None
            try:
                async with self.semaphore:
                    base_port = self.settings.get("source_port_base")
                    source_ports = self.settings.get("source_ports")
                    source_port = source_ports[index] if source_ports else base_port + index if base_port else None
                    socket = await registered(
                        self.url,
                        identity,
                        keepalive=False,
                        source_port=source_port,
                        reset_on_close=source_port is not None,
                    )
                self.connections[index] = socket
                self.backends[index] = socket.response.headers.get("X-Benchmark-Hub")
                # Spread heartbeat phases while retaining Core's five-second period.
                await asyncio.sleep((index % 100) / 100 * self.settings["heartbeat_seconds"])
                while True:
                    waiter = await socket.ping()
                    await asyncio.wait_for(waiter, timeout=5)
                    self.beats[index] = time.monotonic()
                    # websockets reads control frames internally; no application traffic is sent to fleet agents.
                    await asyncio.sleep(self.settings["heartbeat_seconds"])
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - count failures and preserve bounded diagnostic detail
                self.failures += 1
                reason = f"{type(exc).__name__}: {exc}"[:240]
                if reason in self.connection_errors or len(self.connection_errors) < 8:
                    self.connection_errors[reason] = self.connection_errors.get(reason, 0) + 1
            finally:
                self.connections.pop(index, None)
                self.backends.pop(index, None)
                self.beats.pop(index, None)
                if socket:
                    if self.settings.get("source_ports") or self.settings.get("source_port_base"):
                        # Match Core's benchmark: reset the TCP stream before graceful close can enter TIME_WAIT.
                        socket.transport.abort()
                    await socket.close()
            await asyncio.sleep(0.2 + (index % 100) / 100)

    def sample(self):
        """Count live connections and recent confirmed heartbeats instead of reporting offered fleet size alone."""
        now = time.monotonic()
        live = {index for index, socket in self.connections.items() if socket.state is State.OPEN}
        return {
            "at": now,
            "connected": len(live),
            "failures": self.failures,
            "recent_heartbeats": sum(i in live and now - at < 12 for i, at in self.beats.items()),
            "on_a": sum(self.backends.get(i) == "a" for i in live),
            "on_b": sum(self.backends.get(i) == "b" for i in live),
        }

    async def close(self):
        """Cancel registrations and close every owned socket before deployment teardown."""
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


def failed_check(status):
    """Ignore HAProxy's in-progress marker; only a completed failed check begins unhealthy timing."""
    return status.removeprefix("* ").strip() not in ("L7OK", "L4OK", "L6OK", "INI", "")


def qualified_fleet(sample, size, backend):
    """Require the full requested fleet and fresh replies on the specified backend before claiming that scale."""
    return sample["connected"] == sample["recent_heartbeats"] == sample["on_" + backend] == size


def summarize(samples, fault_at, recovered_at, observer_errors=(), *, deadline=None):
    """Measure from first observed unhealthy check; missing or gapped observations cannot prove recovery time."""
    if observer_errors:
        raise RuntimeError("Health observer failed during recovery; timing is not established")
    if deadline is not None and recovered_at >= deadline:
        raise RuntimeError("Recovery evidence arrived after the recovery deadline")
    relevant = [s for s in samples if fault_at <= s["at"] <= recovered_at]
    unhealthy = [s["at"] for s in relevant if failed_check(s.get("check_status", ""))]
    down = [s["at"] for s in relevant if s.get("status", "").startswith("DOWN")]
    observed = [s for s in samples if s["at"] <= recovered_at]
    gaps = [b["at"] - a["at"] for a, b in pairwise(observed) if b["at"] >= fault_at]
    if observed:
        gaps.append(recovered_at - observed[-1]["at"])
    if not unhealthy or not down:
        raise RuntimeError("Missing first-unhealthy or DOWN observations; no recovery time can be established")
    if max(gaps, default=0) > 2:
        raise RuntimeError("Health observation gap exceeded two seconds; repeat this timing measurement")
    return {
        "unhealthy_observed_at": unhealthy[0],
        "down_observed_at": down[0],
        "work_restored_at": recovered_at,
        "recovery_seconds": recovered_at - unhealthy[0],
        "detection_seconds": down[0] - unhealthy[0],
        "post_down_seconds": recovered_at - down[0],
        "fault_to_work_seconds": recovered_at - fault_at,
        "max_observation_gap_seconds": max(gaps, default=0),
    }


async def exercise(settings, project, report):
    """Prove shared-state continuity, establish the fleet, then measure automatic reconnect to the standby."""
    port = settings["port_base"]
    base = f"http://127.0.0.1:{port}"
    fleet = Fleet(settings)
    sockets = []
    tasks = []
    samples = report.setdefault("health_samples", [])
    fleet_samples = report.setdefault("fleet_samples", [])
    resource_samples = report.setdefault("resource_samples", [])
    lag = report.setdefault("generator_lag_seconds", [])

    async def monitor():
        while True:
            began = time.monotonic()
            try:
                raw, _ = await asyncio.to_thread(http, f"http://127.0.0.1:{port + 3}/stats;csv")
                rows = list(csv.DictReader(io.StringIO(raw.decode().lstrip("# "))))
                row = next(r for r in rows if r["pxname"] == "hubs" and r["svname"] == "a")
                received = time.monotonic()
                samples.append({"at": received, "requested": began, **{k: row[k] for k in ("status", "check_status")}})
            except Exception as exc:  # noqa: BLE001 - observer failure is evidence that invalidates timing
                report.setdefault("observer_errors", []).append(str(exc))
            fleet_samples.append(fleet.sample())
            await asyncio.sleep(settings["probe_seconds"])
            lag.append(max(0, time.monotonic() - began - settings["probe_seconds"]))

    async def resources():
        while True:
            try:
                rows = await asyncio.to_thread(project.stats)
                resource_samples.append(
                    {"at": time.monotonic(), "containers": [json.loads(r) for r in rows.splitlines()]}
                )
            except Exception as exc:  # noqa: BLE001 - keep diagnostics failures separate from recovery evidence
                report.setdefault("resource_errors", []).append(str(exc))
            await asyncio.sleep(2)

    try:
        # Runtime /info complements image provenance; keep raw features and build metadata.
        report["hub_info"] = []
        for offset in (1, 2):
            raw, _ = await asyncio.to_thread(http, f"http://127.0.0.1:{port + offset}/info")
            report["hub_info"].append(json.loads(raw))
        sender = await registered(f"ws://127.0.0.1:{port + 2}/ws", "study-sender")
        sockets.append(sender)
        recipient = await registered(f"ws://127.0.0.1:{port}/ws", "study-recipient")
        sockets.append(recipient)
        if recipient.response.headers.get("X-Benchmark-Hub") != "a":
            raise RuntimeError("Initial agent did not connect through primary A")
        payload = b"hub-study-artifact\n" * (settings["artifact_bytes"] // 19 + 1)
        payload = payload[: settings["artifact_bytes"]]
        artifact_url = base + "/storage/study-recipient/checkpoint.bin"
        await asyncio.to_thread(http, artifact_url, payload)
        # A pre-fault read directly through B verifies shared storage before measuring a failover.
        stored, _ = await asyncio.to_thread(http, f"http://127.0.0.1:{port + 2}/storage/study-recipient/checkpoint.bin")
        if stored != payload:
            raise RuntimeError("Shared object storage failed the pre-fault integrity check")
        await send(sender, type="et-send-agent-message", to_agent_id="study-recipient", message={"nonce": project.name})
        status = await until(sender, lambda m: m.get("type") == "et-message-status" and m.get("status") == "delivered")
        pending = await until(recipient, lambda m: m.get("type") == "et-agent-message")
        if pending.get("message_id") != status.get("message_id") or pending.get("message") != {"nonce": project.name}:
            raise RuntimeError("Cross-hub message delivery failed before the fault")
        # Leave this delivery deliberately unacknowledged; B must replay that exact ID after reconnection.
        report["pending_message_id"] = pending["message_id"]
        report["artifact_sha256"] = hashlib.sha256(payload).hexdigest()
        tasks = [asyncio.create_task(monitor()), asyncio.create_task(resources())]
        fleet.start()
        deadline = time.monotonic() + settings["registration_seconds"]
        while True:
            sample = fleet.sample()
            if sample["at"] >= deadline:
                raise RuntimeError(
                    "Requested fleet was not established before the registration deadline; primary was not faulted"
                )
            if qualified_fleet(sample, settings["fleet_size"], "a"):
                break
            print(
                f"Registering: {sample['connected']}/{settings['fleet_size']}; beats={sample['recent_heartbeats']}",
                flush=True,
            )
            await asyncio.sleep(5)
        began = time.monotonic()
        baseline_failures = fleet.failures
        while time.monotonic() - began < settings["warmup_seconds"]:
            sample = fleet.sample()
            if not qualified_fleet(sample, settings["fleet_size"], "a"):
                raise RuntimeError("Fleet did not remain healthy during warmup; primary was not faulted")
            if fleet.failures != baseline_failures or max(lag[-100:], default=0) > 1:
                raise RuntimeError("Warmup connection failures or generator lag; primary was not faulted")
            if report.get("observer_errors") or not samples or samples[-1]["status"] != "UP":
                raise RuntimeError("Primary health observer was not consistently healthy; primary was not faulted")
            # Exercise the same small liveness request used by Core's alive_at_25000 benchmark.
            probe_at = time.monotonic()
            await send(recipient, type="et-alive", timestamp="study")
            await until(recipient, lambda m: m.get("type") == "et-response")
            report.setdefault("alive_probe_seconds", []).append(time.monotonic() - probe_at)
            print(f"Healthy fleet: {sample['connected']}; warmup {int(time.monotonic() - began)}s", flush=True)
            await asyncio.sleep(5)
        report["qualified_fleet"] = fleet.sample()
        if (
            not qualified_fleet(report["qualified_fleet"], settings["fleet_size"], "a")
            or fleet.failures != baseline_failures
            or max(lag[-100:], default=0) > 1
            or report.get("observer_errors")
            or not samples
            or samples[-1]["status"] != "UP"
        ):
            raise RuntimeError("Final full-fleet qualification failed; primary was not faulted")
        print("Pausing primary A; gateway must route reconnects to B", flush=True)
        fault_at = report["fault_at"] = time.monotonic()
        project.pending_pause = asyncio.create_task(asyncio.to_thread(project.pause))
        await asyncio.shield(project.pending_pause)
        deadline = fault_at + settings["recovery_seconds"]
        last_error = "No replacement connection"
        while time.monotonic() < deadline:
            replacement = None
            acknowledged = False
            try:
                replacement = await registered(f"ws://127.0.0.1:{port}/ws", "study-recipient")
                if replacement.response.headers.get("X-Benchmark-Hub") != "b":
                    raise RuntimeError("Gateway has not selected B")
                replay = await until(replacement, lambda m: m.get("type") == "et-agent-message")
                if replay != pending:
                    # Reconnection may add metadata; identity, sender, body and message ID must remain unchanged.
                    for key in ("message_id", "from_agent_id", "message"):
                        if replay.get(key) != pending.get(key):
                            raise RuntimeError("Pending message continuity failed: " + key)
                stored, backend = await asyncio.to_thread(http, artifact_url)
                if stored != payload or backend != "b":
                    raise RuntimeError("Retained artifact was not served intact by B")
                if replacement.state is not State.OPEN:
                    raise RuntimeError("Replacement disconnected before acknowledgement; retry replay")
                acknowledged = True
                await send(replacement, type="et-message-ack", message_id=replay["message_id"])
                await until(
                    replacement,
                    lambda m, message_id=replay["message_id"]: (
                        m.get("type") == "et-message-status"
                        and m.get("status") == "acknowledged"
                        and m.get("message_id") == message_id
                    ),
                )
                await until(
                    sender,
                    lambda m, message_id=replay["message_id"]: (
                        m.get("type") == "et-message-status"
                        and m.get("status") == "acknowledged"
                        and m.get("message_id") == message_id
                    ),
                )
                recovered = time.monotonic()
                report["timing"] = summarize(
                    samples, fault_at, recovered, report.get("observer_errors", ()), deadline=deadline
                )
                report["proof"] = {
                    "backend": backend,
                    "identity": "study-recipient",
                    "message_id": replay["message_id"],
                    "receiver_ack": True,
                    "sender_ack": True,
                    "artifact_intact": True,
                    "primary_paused": project.paused,
                }
                sockets.append(replacement)
                replacement = None
                break
            except Exception as exc:
                if acknowledged:
                    raise RuntimeError(
                        "Recovery evidence failed after acknowledgement; delivery will not be replayed"
                    ) from exc
                last_error = str(exc)
                await asyncio.sleep(0.2)
            finally:
                if replacement:
                    await replacement.close()
        else:
            raise RuntimeError("Recovery was not established within deadline: " + last_error)
        # Keep the fault in place until the requested fleet itself has rejoined B and confirmed heartbeats.
        while time.monotonic() < deadline:
            sample = fleet.sample()
            if qualified_fleet(sample, settings["fleet_size"], "b"):
                restored_at = time.monotonic()
                if restored_at >= deadline:
                    break
                report["restored_fleet"] = sample
                report["fleet_restored_at"] = restored_at
                report["fleet_recovery_seconds"] = (
                    report["fleet_restored_at"] - report["timing"]["unhealthy_observed_at"]
                )
                return
            print(f"Reconnected to B: {sample['on_b']}/{settings['fleet_size']}", flush=True)
            await asyncio.sleep(2)
        raise RuntimeError("Useful work recovered but the complete fleet did not recover within the deadline")
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await fleet.close()
        for socket in sockets:
            await socket.close()
