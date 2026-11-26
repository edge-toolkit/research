"""End-to-end TCP over three IP-in-IP links on one private Docker underlay."""

import json
import subprocess

SERVICES = ("hub", "driver", "rate", "delay")
LINKS = (
    ("driver", "rate", "to-rate", "to-driver", "169.254.200.1/30", "169.254.200.2/30"),
    ("rate", "delay", "to-delay", "to-rate", "169.254.200.5/30", "169.254.200.6/30"),
    ("delay", "hub", "to-hub", "to-delay", "169.254.200.9/30", "169.254.200.10/30"),
)
HUB = "169.254.200.10"
DRIVER = "169.254.200.1"
RETRY_IPS = (DRIVER, "169.254.200.17", "169.254.200.18", "169.254.200.19")


def topology(image, directory, config):
    timeout_env = {}
    if config.get("tolerant_timeouts"):
        policy = config["recovery_policy"]
        timeout_env = {
            "WS_CONNECTION_TIMEOUT": policy["idle_timeout"],
            "HTTP_CLIENT_REQUEST_TIMEOUT": policy["http_request_timeout"],
        }
    common = {
        "image": image,
        "cap_add": ["NET_ADMIN"],
        "pids_limit": 256,
        "volumes": [{"type": "bind", "source": str(directory.resolve()).replace("$", "$$"), "target": "/results"}],
        "networks": ["underlay"],
        "ulimits": {"nofile": {"soft": 65536, "hard": 65536}},
        "sysctls": {"net.ipv4.conf.all.rp_filter": "0", "net.ipv4.conf.default.rp_filter": "0"},
    }
    router = {
        **common,
        "command": ["sleep", "infinity"],
        "mem_limit": config.get("router_memory", "2g"),
        "cpus": config.get("router_cpus", 2),
        "sysctls": {**common["sysctls"], "net.ipv4.ip_forward": "1"},
    }
    return {
        "services": {
            "hub": {
                **common,
                "command": ["et-ws-server"],
                "cpus": 1,
                "mem_limit": config["hub_memory"],
                "environment": {
                    "HTTP_WORKERS": "1",
                    "HTTP_MAX_CONNECTIONS": "30000",
                    "RUST_LOG": "warn",
                    "MODULES_ROOT": "",
                    "WS_AGENT_REGISTRY": "/results/hub-registry.yaml",
                    **timeout_env,
                },
            },
            "driver": {**common, "command": ["sleep", "infinity"], "cpus": 4, "mem_limit": config["driver_memory"]},
            "rate": router,
            "delay": router,
        },
        "networks": {"underlay": {"internal": True}},
    }


def connectivity(invoke, output, stage):
    # A reset close avoids leaving the readiness port in TIME_WAIT between checks.
    code = (
        "import socket,struct,time,json;start=time.monotonic();"
        f"s=socket.create_connection(('{HUB}',8080),timeout=3,source_address=('{DRIVER}',50000));"
        "s.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('ii',1,0));s.close();"
        "print(json.dumps({'seconds':time.monotonic()-start,'connected':True}))"
    )
    try:
        result = invoke("exec", "-T", "driver", "python3", "-c", code, check=False, timeout=15)
    except subprocess.TimeoutExpired as exc:

        def decoded(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value

        evidence = {
            "stage": stage,
            "timeout_seconds": exc.timeout,
            "stdout": decoded(exc.stdout),
            "stderr": decoded(exc.stderr),
        }
        (output / f"connectivity-{stage}.json").write_text(json.dumps(evidence, indent=2) + "\n")
        raise
    evidence = {"stage": stage, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    (output / f"connectivity-{stage}.json").write_text(json.dumps(evidence, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"Routed connectivity failed during {stage}; inspect connectivity and network diagnostics")


def diagnostics(invoke, output, stage):
    result = {}
    for side in SERVICES:
        try:
            value = invoke(
                "exec", "-T", side, "python3", "/usr/local/lib/hub-network/diagnostics.py", timeout=15, check=False
            )
            result[side] = {"status": value.returncode, "stdout": value.stdout, "stderr": value.stderr}
        except Exception as error:  # noqa: BLE001 - preserve other namespaces' diagnostics after a failed exec
            result[side] = {"error": str(error)}
    (output / f"network-diagnostics-{stage}.json").write_text(json.dumps(result, indent=2) + "\n")


def setup(invoke, inspect, ids, case, config, output):
    containers = {side: json.loads(inspect(identifier))[0] for side, identifier in ids.items()}
    addresses = {
        side: next(
            value["IPAddress"]
            for name, value in container["NetworkSettings"]["Networks"].items()
            if name.endswith("_underlay")
        )
        for side, container in containers.items()
    }
    routes = {
        "driver": (HUB, "169.254.200.2", "to-rate"),
        "rate": (HUB, "169.254.200.6", "to-delay"),
        "delay": (DRIVER, "169.254.200.5", "to-rate"),
        "hub": (DRIVER, "169.254.200.9", "to-delay"),
    }
    interfaces = {"rate": ["to-delay", "to-driver"], "delay": ["to-hub", "to-rate"]}
    plan = {
        "hub_address": HUB + ":8080",
        "driver_address": DRIVER,
        "underlay_addresses": addresses,
        "tunnel_links": LINKS,
        "tunnel_mtu": 1400,
        "routes": routes,
        "interfaces": interfaces,
        "path": ["driver", "rate", "delay", "hub"],
        "nat": False,
        "encapsulation": "ipip",
    }
    (output / "network.json").write_text(json.dumps(plan, indent=2) + "\n")
    try:
        for side in SERVICES:
            invoke("exec", "-T", side, "ip", "route", "flush", "default")
        for left, right, left_name, right_name, left_ip, right_ip in LINKS:
            for side, peer, name, inner in ((left, right, left_name, left_ip), (right, left, right_name, right_ip)):
                invoke(
                    "exec",
                    "-T",
                    side,
                    "ip",
                    "tunnel",
                    "add",
                    name,
                    "mode",
                    "ipip",
                    "local",
                    addresses[side],
                    "remote",
                    addresses[peer],
                    "ttl",
                    "64",
                )
                invoke("exec", "-T", side, "ip", "address", "add", inner, "dev", name)
                invoke("exec", "-T", side, "ip", "link", "set", "dev", name, "mtu", "1400", "up")
        for side, (destination, via, interface) in routes.items():
            invoke("exec", "-T", side, "ip", "route", "replace", destination + "/32", "via", via, "dev", interface)
        if config.get("reconnect"):
            for address in RETRY_IPS[1:]:
                invoke("exec", "-T", "driver", "ip", "address", "add", address + "/32", "dev", "to-rate")
                for side, via, interface in (
                    ("rate", DRIVER, "to-driver"),
                    ("delay", "169.254.200.5", "to-rate"),
                    ("hub", "169.254.200.9", "to-delay"),
                ):
                    invoke("exec", "-T", side, "ip", "route", "replace", address + "/32", "via", via, "dev", interface)
            plan["retry_source_addresses"] = RETRY_IPS
            (output / "network.json").write_text(json.dumps(plan, indent=2) + "\n")
        connectivity(invoke, output, "before-shaping")
        for side in ("rate", "delay"):
            invoke(
                "exec",
                "-T",
                side,
                "python3",
                "/usr/local/lib/hub-network/tc_setup.py",
                "configure",
                "--role",
                side,
                "--interfaces",
                *interfaces[side],
                "--agents",
                str(config["agents"]),
                "--timing-probes",
                str(config["timing_probes"]),
                "--rate-kbit",
                str(case["rate_kbit"]),
                "--delay-ms",
                str(case["delay_ms_each_direction"]),
                "--loss-percent",
                str(case["loss_percent"]),
                "--queue-packets",
                str(config["shared_queue_packets"]),
                timeout=900,
            )
        connectivity(invoke, output, "after-shaping")
    except Exception:
        diagnostics(invoke, output, "setup-failure")
        raise
    diagnostics(invoke, output, "ready")
    return plan
