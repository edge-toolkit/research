"""Read-only container CPU, throttling, process accounting and memory-pressure samples."""

import json
import os
import sys
import time
from pathlib import Path


def pairs(path):
    return {key: int(value) for key, value in (line.split() for line in Path(path).read_text().splitlines())}


def host_snapshot():
    """Read the Docker kernel's host counters, including the VM when Docker runs in one."""
    memory = {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.split(":")[0] in ("MemTotal", "MemAvailable")
    }
    cpu = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
    return {
        "total_memory_bytes": memory["MemTotal"],
        "available_memory_bytes": memory["MemAvailable"],
        "cpu_total": sum(cpu),
        "cpu_idle": cpu[3] + cpu[4],
        "swap_out_bytes": pairs("/proc/vmstat")["pswpout"] * os.sysconf("SC_PAGE_SIZE"),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def snapshot():
    cgroup = Path("/sys/fs/cgroup")
    if not all((cgroup / name).is_file() for name in ("cgroup.controllers", "cpu.stat", "cpu.max", "memory.events")):
        raise RuntimeError("Network evaluation requires cgroup v2 CPU and memory accounting")
    processes = []
    for folder in Path("/proc").iterdir():
        if not folder.name.isdecimal():
            continue
        try:
            name = (folder / "comm").read_text().strip()
            if name not in ("et-ws-server", "hub-network-age"):
                continue
            fields = (folder / "stat").read_text().rsplit(")", 1)[1].split()
            ticks = os.sysconf("SC_CLK_TCK")
            processes.append(
                {
                    "pid": int(folder.name),
                    "name": name,
                    "user_seconds": int(fields[11]) / ticks,
                    "system_seconds": int(fields[12]) / ticks,
                }
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
    snmp = Path("/proc/net/snmp").read_text().splitlines()
    tcp = {}
    for header, values in zip(snmp[::2], snmp[1::2], strict=True):
        if header.startswith("Tcp:"):
            tcp = dict(zip(header.split()[1:], map(int, values.split()[1:]), strict=True))
    return {
        "monotonic_seconds": time.monotonic(),
        "cpu_stat": pairs(cgroup / "cpu.stat"),
        "cpu_max": (cgroup / "cpu.max").read_text().strip(),
        "memory_events": pairs(cgroup / "memory.events"),
        "processes": processes,
        "tcp_namespace_counters": tcp,
    }


if __name__ == "__main__":
    print(json.dumps(host_snapshot() if "--host" in sys.argv[1:] else snapshot()))
