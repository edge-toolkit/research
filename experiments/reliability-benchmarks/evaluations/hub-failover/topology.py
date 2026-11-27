"""Isolated primary/standby topology; deployment failover policy lives in this experiment."""

import json
import os
import re
import secrets
import socket
import subprocess
import time
import uuid
from pathlib import Path


def available_source_ports(base, count, *, stop=65536, excluded=()):
    """Allocate checked source ports without touching other applications or host settings."""
    ports = []
    excluded = set(excluded)
    for port in range(base, stop):
        if port in excluded:
            continue
        with socket.socket() as handle:
            try:
                handle.bind(("127.0.0.1", port))
            except OSError:
                continue
        ports.append(port)
        if len(ports) == count:
            return ports
    raise RuntimeError(f"Only {len(ports)} free generator source ports found; {count} required")


def command(args, *, timeout=60):
    """Run a bounded command, preserving a useful diagnostic without dumping credentials."""
    environment = dict(os.environ)
    if args[0] == "docker" and "--host" in args:
        for name in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
            environment.pop(name, None)
    result = subprocess.run(
        list(map(str, args)), capture_output=True, text=True, timeout=timeout, check=False, env=environment
    )
    if result.returncode:
        raise RuntimeError(f"{args[0]} exited {result.returncode}: {result.stderr[-2000:]}")
    return result.stdout


def save(path, data):
    """Atomically publish JSON evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def docker_prefix(target):
    """Use a fixed local endpoint independently of the CLI's mutable default context."""
    if not target or not target.get("host", "").startswith("unix://") or not target.get("id"):
        raise RuntimeError("A recorded local Docker endpoint and daemon identity are required")
    return ["docker", "--host", target["host"]]


def resolve_docker():
    """Resolve environment/context precedence once; this loopback study requires a local daemon."""
    context = os.environ.get("DOCKER_CONTEXT")
    host = os.environ.get("DOCKER_HOST") if not context else None
    if not host:
        context = context or command(["docker", "context", "show"]).strip()
        host = json.loads(command(["docker", "context", "inspect", context]))[0]["Endpoints"]["docker"]["Host"]
    target = {"host": host, "id": "unresolved"}
    info = json.loads(command([*docker_prefix(target), "info", "--format", "{{json .}}"], timeout=15))
    target["id"] = info["ID"]
    docker_prefix(target)
    return target


def verify_daemon(target):
    prefix = docker_prefix(target)
    if command([*prefix, "info", "--format", "{{.ID}}"], timeout=15).strip() != target["id"]:
        raise RuntimeError("Recorded Docker daemon identity changed; cleanup remains unverified")


def verify_absent(project, target):
    """Verify every owned resource inventory; a failed inventory never proves teardown."""
    if not re.fullmatch(r"et-hub-study-[0-9a-f]{12}", project):
        raise RuntimeError("Unknown Docker project ownership")
    verify_daemon(target)
    errors = []
    for resource, args in (
        ("containers", ["ps", "-aq"]),
        ("volumes", ["volume", "ls", "-q"]),
        ("networks", ["network", "ls", "-q"]),
    ):
        try:
            if command(
                [*docker_prefix(target), *args, "--filter", "label=com.docker.compose.project=" + project]
            ).strip():
                errors.append("Owned " + resource + " remain")
        except Exception as exc:  # noqa: BLE001 - check the remaining inventories even after one fails
            errors.append(resource + " inventory: " + str(exc))
    # Inventories are separate requests, so these identity checks cannot be atomic with them.
    verify_daemon(target)
    return errors


def gateway():
    """Actively detect primary loss, close stale tunnels, and route reconnects to the warm standby."""
    return Path(__file__).with_name("haproxy.cfg").read_text()


def document(settings, image, directory, secret):
    """Share one JetStream mailbox and object store; bound each hub separately to reserve standby resources."""
    port = settings["port_base"]
    services = {
        "nats": {
            "image": settings["nats_image"],
            "command": ["--jetstream", "--store_dir", "/data", "--http_port", "8222"],
            "ports": [f"127.0.0.1:{port + 4}:8222"],
            "volumes": ["nats:/data"],
            "cpus": 1,
            "mem_limit": "1g",
        },
        "storage": {
            "image": settings["storage_image"],
            "user": "0:0",
            "entrypoint": ["sh", "-c", "mkdir -p /data/et-storage && exec /entrypoint.sh rustfs"],
            "environment": {"RUSTFS_ACCESS_KEY": "study", "RUSTFS_SECRET_KEY": secret, "RUSTFS_VOLUMES": "/data"},
            "volumes": ["storage:/data"],
            "ports": [f"127.0.0.1:{port + 5}:9000"],
            "cpus": 1,
            "mem_limit": "1g",
        },
        "gateway": {
            "image": settings["gateway_image"],
            "cpus": 2,
            "mem_limit": "1g",
            "volumes": [str(directory / "haproxy.cfg") + ":/usr/local/etc/haproxy/haproxy.cfg:ro"],
            "ports": [f"127.0.0.1:{port}:8080", f"127.0.0.1:{port + 3}:8404"],
            "ulimits": {"nofile": {"soft": 131072, "hard": 131072}},
        },
    }
    for offset, name in enumerate(("hub-a", "hub-b"), 1):
        services[name] = {
            "image": image,
            "cpus": settings["hub_cpus"],
            "mem_limit": settings["hub_memory"],
            "ports": [f"127.0.0.1:{port + offset}:8080"],
            "ulimits": {"nofile": {"soft": 131072, "hard": 131072}},
            "environment": {
                "WS_BACKPLANE": "nats://nats:4222",
                "WS_MESSAGE_RETENTION": "all",
                "HTTP_WORKERS": "1",
                "HTTP_MAX_CONNECTIONS": str(settings.get("max_connections", 30000)),
                "WS_CONNECTION_TIMEOUT": "30s",
                "STORAGE_URL": "s3://et-storage",
                "AWS_ACCESS_KEY_ID": "study",
                "AWS_SECRET_ACCESS_KEY": secret,
                "AWS_ENDPOINT": "http://storage:9000",
                "AWS_ALLOW_HTTP": "true",
                "AWS_DEFAULT_REGION": "us-east-1",
                "MODULES_ROOT": "",
            },
        }
    # Compose's unique project label scopes all resources and avoids touching existing deployments.
    return {"services": services, "volumes": {"nats": {}, "storage": {}}}


class Deployment:
    """Own a fresh project per attempt, with no automatic primary restart during measurement."""

    def __init__(self, settings, image, output, *, docker_target):
        self.docker_target = docker_target
        self.docker = docker_prefix(docker_target)
        self.output = Path(output).resolve()
        self.name = "et-hub-study-" + uuid.uuid4().hex[:12]
        self.directory = self.output / "private"
        self.directory.mkdir(mode=0o700)
        self.secret = secrets.token_hex(24)
        (self.directory / "haproxy.cfg").write_text(gateway())
        save(self.directory / "compose.json", document(settings, image, self.directory, self.secret))
        self.prefix = [*self.docker, "compose", "-p", self.name, "-f", self.directory / "compose.json"]
        self.paused = False
        self.collection_errors = []
        self.cleanup_diagnostics = []
        self.pending_pause = None

    def compose(self, *args, timeout=120):
        """Execute commands only in this attempt's randomly named project."""
        return command([*self.prefix, *args], timeout=timeout)

    def stats(self):
        """Sample CPU, memory, network and process counts for all owned containers."""
        ids = self.compose("ps", "-q").split()
        return command([*self.docker, "stats", "--no-stream", "--format", "{{json .}}", *ids]) if ids else ""

    def pause(self):
        """Hold the primary unavailable so any recovery must use the alternate hub."""
        self.compose("pause", "hub-a")
        self.paused = True

    def cleanup(self):
        """Collect diagnostics and attempt teardown even if log collection or unpause fails."""
        errors = []
        try:
            verify_daemon(self.docker_target)
        except Exception as exc:  # noqa: BLE001 - unreachable or replaced daemons cannot prove teardown
            return [str(exc)]
        try:
            self._logs()
        except Exception as exc:  # noqa: BLE001 - missing diagnostics do not imply leaked resources
            self.collection_errors.append(str(exc))
        for action in (self._unpause, lambda: self.compose("down", "--volumes", "--timeout", "5")):
            try:
                action()
            except Exception as exc:  # noqa: BLE001 - attempt every cleanup action and retain all failures
                self.cleanup_diagnostics.append(str(exc))
        errors.extend(verify_absent(self.name, self.docker_target))
        # Never export the Compose document containing credentials.
        if not errors:
            (self.directory / "compose.json").unlink(missing_ok=True)
        return errors

    def _logs(self):
        """Retain timestamped logs without the generated object-store secret."""
        logs = self.compose("logs", "--no-color", "--timestamps")
        (self.output / "containers.log").write_text(logs.replace(self.secret, "[redacted]"))

    def _unpause(self):
        """Release an owned paused container before removing it."""
        if self.paused:
            self.compose("unpause", "hub-a")
            self.paused = False


def preflight(settings, snapshot, *, docker_target=None):
    """Reject stale images, insufficient descriptors/resources, and occupied ports before starting a fleet."""
    import resource
    import socket

    target = resolve_docker() if docker_target is None else docker_target
    docker = docker_prefix(target)
    info = json.loads(command([*docker, "info", "--format", "{{json .}}"], timeout=15))
    if info["ID"] != target["id"]:
        raise RuntimeError("Docker daemon identity changed before startup")
    if info["MemTotal"] < 8 * 1024**3:
        raise RuntimeError("Docker needs at least 8 GiB for this isolated two-hub study")
    if info["NCPU"] < settings["hub_cpus"] * 2 + 4:
        raise RuntimeError("Insufficient Docker CPUs for both hubs, broker, storage and gateway budgets")
    needed = settings["fleet_size"] + 512
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < needed:
        raise RuntimeError(f"Open-file limit {soft} is below {needed}; raise ulimit -n before running (hard={hard})")
    metadata = {}
    for image in [snapshot["image"], *[v for k, v in settings.items() if k.endswith("_image")]]:
        row = json.loads(command([*docker, "image", "inspect", image]))[0]
        metadata[image] = {"id": row["Id"], "architecture": row["Architecture"], "digests": row.get("RepoDigests")}
    hub = json.loads(command([*docker, "image", "inspect", snapshot["image"]]))[0]
    if hub["Id"] != snapshot["image_id"]:
        raise RuntimeError("Shared Core release changed; run prepare again")
    for port in range(settings["port_base"], settings["port_base"] + 6):
        with socket.socket() as handle:
            handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            handle.bind(("127.0.0.1", port))
    report = {
        "docker_target": target,
        "docker": {k: info[k] for k in ("ID", "ServerVersion", "NCPU", "MemTotal", "Architecture")},
        "images": metadata,
        "checked_at": time.time(),
    }
    if settings.get("source_port_base"):
        import platform

        if platform.system() == "Linux":
            ephemeral = list(map(int, Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split()))
        elif platform.system() == "Darwin":
            ephemeral = [
                int(command(["/usr/sbin/sysctl", "-n", "net.inet.ip.portrange." + key])) for key in ("first", "last")
            ]
        else:
            raise RuntimeError("Checked source-port allocation supports Linux and macOS")
        excluded = {*range(ephemeral[0], ephemeral[1] + 1), *range(settings["port_base"], settings["port_base"] + 6)}
        report["source_ports"] = available_source_ports(
            settings["source_port_base"],
            settings["fleet_size"],
            stop=settings.get("source_port_limit", 65536),
            excluded=excluded,
        )
        report["host_ephemeral_port_range"] = ephemeral
    return report
