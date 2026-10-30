"""Best-effort O2 ingestion with local replayable evidence and bounded requests."""

from __future__ import annotations

import argparse
import base64
import hashlib
import ipaddress
import json
import os
import re
import time
from dataclasses import dataclass, field
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from et_experiments.serialization import json_dumps, json_safe

from .o2_records import records


@dataclass(frozen=True)
class O2Config:
    """Operator settings; credentials never come from browser requests or catalogs."""

    url: str
    organization: str
    stream: str
    authorization: str = field(repr=False)

    @classmethod
    def from_environment(cls):
        """Disable ingestion unless a URL is explicitly configured."""
        url = os.environ.get("BENCHMARK_O2_URL", "").rstrip("/")
        if not url:
            return None
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("BENCHMARK_O2_URL must be an HTTP(S) base URL without credentials, query or fragment")
        loopback = parsed.hostname == "localhost"
        try:
            loopback = loopback or ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            pass  # DNS names are not assumed to resolve to a trusted local host.
        if parsed.scheme == "http" and not loopback and os.environ.get("BENCHMARK_O2_ALLOW_HTTP") != "1":
            raise ValueError("use HTTPS for remote O2 or explicitly allow trusted-network HTTP")
        organization = os.environ.get("BENCHMARK_O2_ORG", "default")
        stream = os.environ.get("BENCHMARK_O2_STREAM", "reliability_benchmarks")
        if not all(re.fullmatch(r"[A-Za-z0-9_-]+", item) for item in (organization, stream)):
            raise ValueError("O2 organization and stream must contain only letters, digits, underscores or hyphens")
        user, password = os.environ.get("BENCHMARK_O2_USER", ""), os.environ.get("BENCHMARK_O2_PASSWORD", "")
        if not user or not password or ":" in user:
            raise ValueError("set BENCHMARK_O2_USER and BENCHMARK_O2_PASSWORD for an O2 ingestion identity")
        authorization = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
        return cls(url, organization, stream, authorization)

    @property
    def endpoint(self):
        """Address the selected stream on the existing Core-deployed O2 service."""
        return f"{self.url}/api/{self.organization}/{self.stream}/_json"


class NoRedirect(HTTPRedirectHandler):
    """Never forward ingestion credentials through an HTTP redirect."""

    def redirect_request(self, _req, _fp, _code, _msg, _headers, _newurl):
        return None


def deliver(path: Path, config: O2Config, budget_seconds: float = 12) -> dict:
    """Resume acknowledged batches; ambiguous failures may be replayed at least once."""
    payload = path.read_bytes()
    items = json.loads(payload)
    digest = hashlib.sha256(payload).hexdigest()
    receipt_path = path.with_suffix(".receipt.json")
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    same = receipt.get("sha256") == digest and receipt.get("endpoint") == config.endpoint
    sent = receipt.get("sent", 0) if same else 0
    if type(sent) is not int or not 0 <= sent <= len(items):
        raise ValueError("invalid O2 delivery receipt")
    opener = build_opener(ProxyHandler({}), NoRedirect())
    deadline = time.monotonic() + budget_seconds
    while sent < len(items):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("O2 export budget exceeded; replay the local outbox later")
        batch = items[sent : sent + 200]
        request = Request(
            config.endpoint,
            data=json_dumps(batch).encode(),
            headers={"Content-Type": "application/json", "Authorization": config.authorization},
            method="POST",
        )
        with opener.open(request, timeout=min(3, remaining)) as response:
            acknowledgement = json.loads(response.read(65536))
        if not isinstance(acknowledgement, dict):
            raise TypeError("O2 returned a malformed acknowledgement")
        statuses = acknowledgement.get("status", [])
        if not isinstance(statuses, list) or not all(
            isinstance(item, dict)
            and item.get("name") == config.stream
            and all(type(item.get(key)) is int and item[key] >= 0 for key in ("successful", "failed"))
            for item in statuses
        ):
            raise ValueError("O2 returned malformed ingestion counts")
        accepted = sum(item.get("successful", 0) for item in statuses)
        failed = sum(item.get("failed", 0) for item in statuses)
        if acknowledgement.get("code") != 200 or failed or accepted != len(batch):
            raise ValueError("O2 did not acknowledge every record; batch retained for replay")
        sent += len(batch)
        receipt = {"sha256": digest, "endpoint": config.endpoint, "sent": sent}
        temporary = receipt_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
        temporary.replace(receipt_path)
    return {"status": "exported", "records": sent, "stream": config.stream, "outbox": str(path)}


def publish_report(report: dict, output: Path, context: dict | None = None) -> dict:
    """Preserve benchmark outcomes even when O2 is unavailable or misconfigured."""
    path = None
    try:
        config = O2Config.from_environment()
        if config is None:
            return {"status": "disabled"}
        items = records(json_safe(report), json_safe(context))
        # Hash the job identity instead of trusting it as an operator-side path.
        labels = {key: items[0][key] for key in ("job_id", "controller_run_id", "repetition")}
        identity = hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest()[:24]
        directory = output / "o2"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{identity}.json"
        payload = json_dumps(items, sort_keys=True) + "\n"
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(path)
        return deliver(path, config)
    # Optional export must not change benchmark outcomes, including for malformed adapter data.
    except Exception as exc:  # noqa: BLE001
        # Do not expose server response bodies or URLs containing secrets to the UI.
        detail = f"HTTP {exc.code}" if isinstance(exc, HTTPError) else type(exc).__name__
        return {"status": "unavailable", "reason": detail, "outbox": str(path) if path else None}


def main() -> int:
    """Export an existing report or retry one locally retained O2 outbox."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--report", type=Path)
    group.add_argument("--replay", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--application", default="unspecified")
    args = parser.parse_args()
    try:
        config = O2Config.from_environment()
        if config is None:
            parser.error("set BENCHMARK_O2_URL and ingestion credentials first")
        result = (
            deliver(args.replay, config)
            if args.replay
            else publish_report(
                json.loads(args.report.read_text()),
                args.output or args.report.parent,
                {"application": args.application, "job_id": args.report.stem},
            )
        )
    except (OSError, URLError, HTTPException, ValueError, TypeError) as exc:
        result = {"status": "unavailable", "reason": type(exc).__name__}
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "exported" else 1


if __name__ == "__main__":
    raise SystemExit(main())
