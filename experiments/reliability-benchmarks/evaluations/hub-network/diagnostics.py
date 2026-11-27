"""Read-only evidence from an owned network namespace; no host firewall or sysctl changes."""

import json
import subprocess
import time
from pathlib import Path


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError as error:
        return {"error": str(error)}


def main():
    result = {}
    deadline = time.monotonic() + 12
    for name, arguments in {
        "addresses": ["ip", "-j", "address", "show"],
        "links": ["ip", "-j", "-s", "link", "show"],
        "routes": ["ip", "-j", "route", "show", "table", "all"],
        "neighbours": ["ip", "-j", "neighbour", "show"],
        "tunnels": ["ip", "-j", "tunnel", "show"],
    }.items():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            result[name] = {"status": None, "error": "Diagnostic budget exhausted"}
            continue
        try:
            value = subprocess.run(arguments, capture_output=True, text=True, check=False, timeout=min(2, remaining))
            result[name] = {"status": value.returncode, "stdout": value.stdout, "stderr": value.stderr}
        except (subprocess.TimeoutExpired, OSError) as error:
            result[name] = {"status": None, "error": str(error)}
    result["tcp"] = read("/proc/net/tcp")
    result["ip_counters"] = read("/proc/net/snmp")
    result["sysctls"] = {
        str(path): read(path)
        for path in [Path("/proc/sys/net/ipv4/ip_forward"), *Path("/proc/sys/net/ipv4/conf").glob("*/rp_filter")]
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
