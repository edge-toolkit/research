"""Generate bounded, paced UDP contention against a private receiver and measure echoed probes."""

from __future__ import annotations

import datetime as dt
import json
import select
import socket
import struct
import sys
import time


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def serve(options):
    """Echo token-validated probe headers for a bounded lifetime, keeping responses small."""
    token = options["token"].encode()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as peer:
        peer.bind(("0.0.0.0", options["port"]))
        peer.settimeout(1)
        deadline = time.monotonic() + options["lifetime_seconds"]
        while time.monotonic() < deadline:
            try:
                packet, address = peer.recvfrom(1500)
            except TimeoutError:
                continue
            if packet.startswith(token) and len(packet) >= len(token) + 12:
                peer.sendto(packet[: len(token) + 12], address)


def stage(peer, token, rate_kbit, duration, drain_seconds=1):
    """Record achieved offered load, echoed payload-equivalent goodput, RTT and unacknowledged probes."""
    started_at = timestamp()
    start = time.monotonic()
    stop = start + duration
    interval = 1200 * 8 / (rate_kbit * 1000)
    next_send = start
    sent, received, delays = {}, set(), []
    local_send_failures = 0
    while time.monotonic() < stop + drain_seconds:
        now = time.monotonic()
        if now < stop and now >= next_send:
            identity = len(sent)
            packet = token + struct.pack("!Q", identity)
            try:
                peer.send(packet + bytes(1200 - len(packet)))
            except BlockingIOError:
                local_send_failures += 1
            else:
                sent[identity] = now
            next_send = now + interval
        ready, _, _ = select.select([peer], [], [], 0.001)
        if ready:
            reply = peer.recv(1500)
            if reply.startswith(token) and len(reply) == len(token) + 8:
                identity = struct.unpack("!Q", reply[len(token) :])[0]
                if identity in sent and identity not in received:
                    received.add(identity)
                    delays.append((time.monotonic() - sent[identity]) * 1000)
    elapsed = time.monotonic() - start
    delays.sort()
    return {
        "started_at": started_at,
        "finished_at": timestamp(),
        "requested_kbit": rate_kbit,
        "offered_kbit": len(sent) * 1200 * 8 / duration / 1000,
        "goodput_kbit": len(received) * 1200 * 8 / elapsed / 1000,
        "sent_packets": len(sent),
        "local_send_failures": local_send_failures,
        "acknowledged_packets": len(received),
        "ack_loss_ratio": 1 - len(received) / max(1, len(sent)),
        "rtt_p95_ms": delays[round((len(delays) - 1) * 0.95)] if delays else None,
    }


def run_profile(options):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as peer:
        peer.connect((options["host"], options["port"]))
        peer.setblocking(False)
        stages = []
        for index, multiplier in enumerate(options["levels"]):
            # Stage-specific tokens prevent delayed replies from satisfying a later measurement.
            token = options["token"].encode() + struct.pack("!I", index)
            measured = stage(peer, token, options["rate_kbit"] * multiplier, options["hold_seconds"])
            measured["offered_multiplier"] = multiplier
            stages.append(measured)
    baseline, peak = stages[0], stages[-1]
    degraded = peak["ack_loss_ratio"] > 0.01 or (
        peak["rtt_p95_ms"] is not None
        and baseline["rtt_p95_ms"] is not None
        and peak["rtt_p95_ms"] > max(2 * baseline["rtt_p95_ms"], baseline["rtt_p95_ms"] + 5)
    )
    plateau = peak["goodput_kbit"] <= max(s["goodput_kbit"] for s in stages[:-1]) * 1.2
    baseline_healthy = baseline["acknowledged_packets"] > 0 and baseline["ack_loss_ratio"] < 0.1
    delivered = max(s["goodput_kbit"] for s in stages) >= options["rate_kbit"] * 0.5
    observed = baseline_healthy and delivered and peak["offered_kbit"] > options["rate_kbit"] and degraded and plateau
    return {
        "status": "restored" if observed else "not-observed",
        "stages": stages,
        "metrics": {
            "network_offered_peak_kbit": max(s["offered_kbit"] for s in stages),
            "network_goodput_peak_kbit": max(s["goodput_kbit"] for s in stages),
            "network_ack_loss_peak_ratio": max(s["ack_loss_ratio"] for s in stages),
            "network_rtt_p95_peak_ms": max(
                (s["rtt_p95_ms"] for s in stages if s["rtt_p95_ms"] is not None), default=None
            ),
            "pressure_observed": int(observed),
        },
        "measurement_note": "A controlled client egress bottleneck, not physical-link capacity. Goodput is "
        "payload-equivalent delivery confirmed by small echo replies; loss is round-trip.",
    }


def probe(options):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as peer:
        peer.settimeout(1)
        message = options["token"].encode() + bytes(12)
        peer.sendto(message, ("127.0.0.1", options["port"]))
        reply, _address = peer.recvfrom(1500)
        if reply != message:
            raise ValueError("invalid private receiver response")


if __name__ == "__main__":
    options = json.loads(sys.argv[2])
    if sys.argv[1] == "serve":
        serve(options)
    elif sys.argv[1] == "probe":
        probe(options)
    else:
        print(json.dumps(run_profile(options)), flush=True)
