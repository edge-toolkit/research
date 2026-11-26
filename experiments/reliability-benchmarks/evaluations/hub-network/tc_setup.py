"""Traffic shaping in dedicated rate and impairment routers; never on the Core hub."""

import argparse
import gzip
import json
import math
import subprocess
from pathlib import Path


def rate_rules(ports, side, rate_kbit):
    if not rate_kbit:
        return ["qdisc replace dev DEVICE root pfifo limit 65536"]
    field = "sport" if side == "driver" else "dport"
    mask = "0x00ff0000" if side == "driver" else "0x000000ff"
    commands = [
        "qdisc replace dev DEVICE root handle 1: htb default 1",
        "class add dev DEVICE parent 1: classid 1:1 htb rate 10gbit ceil 10gbit quantum 1514",
        "filter add dev DEVICE parent 1: protocol ip pref 10 handle 100: u32 divisor 256",
        (
            "filter add dev DEVICE parent 1: protocol ip pref 9 u32 match ip protocol 6 0xff "
            f"match ip ihl 5 0xf hashkey mask {mask} at 20 link 100:"
        ),
    ]
    for index, port in enumerate(ports):
        identifier = f"{index + 10:x}"
        commands.extend(
            [
                (
                    f"class add dev DEVICE parent 1: classid 1:{identifier} htb rate {rate_kbit}kbit "
                    f"ceil {rate_kbit}kbit burst 1600 cburst 1600 quantum 1514"
                ),
                f"qdisc add dev DEVICE parent 1:{identifier} handle {identifier}: pfifo limit 64",
                (
                    f"filter add dev DEVICE parent 1: protocol ip pref 10 u32 ht 100:{port & 255:x}: "
                    f"match ip {field} {port} 0xffff flowid 1:{identifier}"
                ),
            ]
        )
    return commands


def delay_rules(delay_ms, loss_percent, queue_packets):
    commands = [
        "qdisc replace dev DEVICE root handle 1: prio bands 3 priomap " + " ".join(["1"] * 16),
        (
            "filter add dev DEVICE parent 1: protocol ip pref 1 u32 match ip protocol 6 0xff "
            "match ip sport 50001 0xffff flowid 1:1"
        ),
        (
            "filter add dev DEVICE parent 1: protocol ip pref 2 u32 match ip protocol 6 0xff "
            "match ip dport 50001 0xffff flowid 1:1"
        ),
        "filter add dev DEVICE parent 1: protocol ip pref 10 u32 match ip protocol 6 0xff flowid 1:3",
    ]
    commands[1:1] = [
        (
            f"filter add dev DEVICE parent 1: protocol ip pref {priority} u32 match ip protocol 6 0xff "
            f"match ip {field} 50000 0xffff flowid 1:1"
        )
        for priority, field in ((3, "sport"), (4, "dport"))
    ]
    netem = f"qdisc add dev DEVICE parent 1:3 handle 30: netem limit {queue_packets}"
    if delay_ms:
        netem += f" delay {delay_ms}ms"
    if loss_percent:
        netem += f" loss random {loss_percent}%"
    return [*commands, netem]


def state(interface):
    result = {}
    for kind in ("qdisc", "class", "filter"):
        if kind == "filter" and not any(row["kind"] in ("htb", "prio") for row in result["qdisc"]):
            result[kind] = []
            continue
        args = ["tc", "-j", "-s", kind, "show", "dev", interface]
        if kind == "filter":
            args += ["parent", "1:"]
        result[kind] = json.loads(subprocess.check_output(args))
    return result


def verify(data, role, ports, rate_kbit, delay_ms, loss_percent, queue_packets):
    if role == "rate" and rate_kbit:
        classes = [row for row in data["class"] if row.get("kind", row.get("class")) == "htb"]
        filters = [row for row in data["filter"] if row.get("options", {}).get("flowid")]
        if len(classes) != len(ports) + 1 or len(filters) != len(ports):
            raise RuntimeError("Live per-vessel rate-class/filter count mismatch")
        for index in sorted({0, len(ports) // 2, len(ports) - 1}):
            identifier = f"1:{index + 10:x}"
            row = next(row for row in classes if row.get("handle", row.get("classid")) == identifier)
            actual = row.get("options", {}).get("rate", row.get("rate"))
            if actual != 1250:
                raise RuntimeError(f"Class {identifier} rate {actual}, expected 1250 bytes/s")
        if any(row["kind"] == "netem" for row in data["qdisc"]):
            raise RuntimeError("Delay queues must not be nested under per-vessel rate classes")
    elif role == "delay":
        queues = [row for row in data["qdisc"] if row["kind"] == "netem"]
        if len(queues) != 1:
            raise RuntimeError("Expected one shared impairment queue per direction")
        options = queues[0]["options"]
        actual_delay = options.get("delay", {}).get("delay", 0)
        actual_loss = options.get("loss-random", {}).get("loss", 0)
        if (
            not math.isclose(actual_delay, delay_ms / 1000, abs_tol=1e-6)
            or not math.isclose(actual_loss, loss_percent / 100, abs_tol=1e-6)
            or options["limit"] != queue_packets
        ):
            raise RuntimeError("Live impairment parameters do not match the case")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["configure", "change", "collect"])
    parser.add_argument("--role", choices=["rate", "delay"], required=True)
    parser.add_argument("--interfaces", nargs=2, required=True, help="toward hub, toward driver")
    parser.add_argument("--output", type=Path, default=Path("/results"))
    parser.add_argument("--agents", type=int, default=25000)
    parser.add_argument("--timing-probes", type=int, default=8)
    parser.add_argument("--rate-kbit", type=int, default=0)
    parser.add_argument("--delay-ms", type=int, default=0)
    parser.add_argument("--loss-percent", type=int, default=0)
    parser.add_argument("--queue-packets", type=int, default=65536)
    args = parser.parse_args()
    if not 1 <= args.agents <= 25000 or not 1 <= args.timing_probes <= 32:
        raise ValueError("Invalid fleet/probe count")
    if args.rate_kbit not in (0, 10) or args.delay_ms not in (0, 250, 500) or args.loss_percent not in (0, 30):
        raise ValueError("Parameters outside declared study")
    ports = [*range(20000, 20000 + args.agents), *range(45000, 45000 + args.timing_probes)]
    result = {}
    for interface, side in zip(args.interfaces, ("driver", "hub"), strict=True):
        if args.action in ("configure", "change"):
            if args.action == "change" and args.role != "delay":
                raise ValueError("Only impairment queues can be changed live")
            commands = (
                rate_rules(ports, side, args.rate_kbit)
                if args.role == "rate"
                else delay_rules(args.delay_ms, args.loss_percent, args.queue_packets)
            )
            if args.action == "change":
                commands = [commands[-1].replace("qdisc add", "qdisc change", 1)]
            batch = "\n".join(commands).replace("DEVICE", interface) + "\n"
            suffix = "-change" if args.action == "change" else ""
            (args.output / f"{args.role}-{side}-tc{suffix}.batch").write_text(batch)
            offload = subprocess.run(
                ["ethtool", "-K", interface, "gso", "off", "tso", "off", "gro", "off"],
                capture_output=True,
                text=True,
                check=False,
            )
            subprocess.run(["tc", "-batch", "-"], input=batch, text=True, check=True)
            result[interface] = state(interface)
            result[interface]["offload"] = {"status": offload.returncode, "output": offload.stdout + offload.stderr}
            verify(
                result[interface],
                args.role,
                ports,
                args.rate_kbit,
                args.delay_ms,
                args.loss_percent,
                args.queue_packets,
            )
        else:
            result[interface] = state(interface)
    with gzip.open(args.output / f"{args.role}-tc-{args.action}.json.gz", "wt") as stream:
        json.dump(result, stream)
    print(json.dumps({"role": args.role, "interfaces": args.interfaces, "verified": args.action != "collect"}))


if __name__ == "__main__":
    main()
