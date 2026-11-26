"""Validate completed FL rounds and summarise measured, single-host scaling evidence."""

import ast
import csv
import datetime as dt
import math
import re
import statistics
from decimal import Decimal

ROUND = re.compile(r"coordinator completed round (\d+): (\{.*\})")
CLIENT = re.compile(r"(vessel_\d+) completed round (\d+): (\{.*\})")


def metric_interval(item):
    """Normalize the Metrics API's duration while preserving both measurement interval boundaries."""
    window = item["window"]
    if not re.fullmatch(r"(?:\d+(?:\.\d+)?(?:ns|us|µs|ms|s|m|h))+", window):
        raise ValueError("Invalid Metrics API window: " + window)
    units = {
        "ns": Decimal("1e-9"),
        "us": Decimal("1e-6"),
        "µs": Decimal("1e-6"),
        "ms": Decimal(".001"),
        "s": Decimal(1),
        "m": Decimal(60),
        "h": Decimal(3600),
    }
    seconds = sum(
        Decimal(value) * units[unit] for value, unit in re.findall(r"(\d+(?:\.\d+)?)(ns|us|µs|ms|s|m|h)", window)
    )
    if seconds <= 0:
        raise ValueError("Metrics API windows must have positive duration")
    return dt.datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")), seconds


def overlaps_training(item, began, ended):
    end, seconds = metric_interval(item)
    return end > began and end - dt.timedelta(seconds=float(seconds)) < ended


def quantities(value, memory=False):
    """Convert Kubernetes CPU quantities to cores and memory quantities to MiB."""
    scales = {
        "n": Decimal("1e-9"),
        "u": Decimal("1e-6"),
        "m": Decimal("1e-3"),
        **{suffix: Decimal(1024) ** power for power, suffix in enumerate(("Ki", "Mi", "Gi", "Ti", "Pi", "Ei"), 1)},
        **{suffix: Decimal(1000) ** power for power, suffix in enumerate(("k", "M", "G", "T", "P", "E"), 1)},
    }
    for suffix in sorted(scales, key=len, reverse=True):
        if value.endswith(suffix):
            result = Decimal(value[: -len(suffix)]) * scales[suffix]
            break
    else:
        result = Decimal(value)
    return float(result / (1024**2) if memory else result)


def parse_rounds(logs, adapter=None, expected_rounds=0):
    """Keep one completion per round, rejecting duplicate/conflicting completion records."""
    rounds, clients = {}, {}
    for name, content in logs.items():
        pattern = ROUND if name == "vessel-coordinator" else CLIENT
        for line in content.splitlines():
            match = pattern.search(line)
            if not match:
                continue
            index = int(match.group(1 if name == "vessel-coordinator" else 2))
            metrics = ast.literal_eval(match.group(2 if name == "vessel-coordinator" else 3))
            target = rounds if name == "vessel-coordinator" else clients.setdefault(match.group(1), {})
            if index in target:
                raise ValueError("Duplicate round completion: " + name)
            target[index] = metrics
    if adapter is not None:
        observed = adapter.parse_logs(logs, expected_rounds)
        parsed_rounds = {row["round"]: row["metrics"] for row in observed["rounds"]}
        parsed_clients = {
            name: {row["round"]: row["metrics"] for row in values} for name, values in observed["client_rounds"].items()
        }
        if parsed_rounds != rounds or parsed_clients != clients:
            raise ValueError("Harbour-FL adapter and recorded round evidence disagree")
    return rounds, clients


def resource_summary(samples, expected_roles):
    """Deduplicate Metrics API windows, retain role coverage and label window-averaged CPU explicitly."""
    seen = set()
    per_role = {name: [] for name in expected_roles}
    totals = {}
    for sample in samples:
        for pod in sample.get("items", []):
            if not pod.get("containers"):
                continue
            podname = pod["metadata"]["name"]
            role = next((r for r in expected_roles if podname.startswith(r + "-")), None)
            if role is None:
                continue
            interval = metric_interval(pod)
            key = (podname, *interval)
            if key in seen:
                continue
            seen.add(key)
            cpu = sum(quantities(c["usage"]["cpu"]) for c in pod["containers"])
            memory = sum(quantities(c["usage"]["memory"], True) for c in pod["containers"])
            per_role[role].append((cpu, memory))
            # Aggregate matching Metrics API timestamps; partial windows are retained separately.
            totals.setdefault(interval, {})[role] = (cpu, memory)
    missing = [role for role, values in per_role.items() if not values]
    if missing:
        raise ValueError("Missing resource samples for: " + ", ".join(missing))
    complete = [list(values.values()) for values in totals.values() if set(values) == set(expected_roles)]
    # Pod scrape timestamps usually differ slightly, so report sum-of-role-means instead of inventing synchrony.
    return {
        "cpu_definition": (
            "Metrics API window-average CPU cores; untrimmed windows can include startup and idle time; "
            "memory is working-set MiB"
        ),
        "aggregate_mean_cpu_cores": sum(statistics.mean(v[0] for v in values) for values in per_role.values()),
        "aggregate_mean_memory_mib": sum(statistics.mean(v[1] for v in values) for values in per_role.values()),
        "synchronised_complete_windows": len(complete),
        "roles": {
            role: {
                "windows": len(values),
                "mean_cpu_cores": statistics.mean(v[0] for v in values),
                "peak_window_cpu_cores": max(v[0] for v in values),
                "mean_memory_mib": statistics.mean(v[1] for v in values),
                "peak_memory_mib": max(v[1] for v in values),
            }
            for role, values in per_role.items()
        },
    }


def summarise(logs, samples, partition, settings, adapter=None):
    """Require all clients and all nominal samples before a scaling point can be resumed as successful."""
    rounds, clients = parse_rounds(logs, adapter, settings["rounds"])
    expected_rounds = set(range(settings["rounds"]))
    expected_clients = set(partition["clients"])
    if set(rounds) != expected_rounds or set(clients) != expected_clients:
        raise ValueError("Incomplete coordinator/client rounds")
    records = []
    for index in sorted(rounds):
        metrics = rounds[index]
        if metrics["participating_clients"] != len(expected_clients):
            raise ValueError("Degraded participation is not a valid scaling point")
        if not math.isfinite(metrics["round_work_seconds"]) or metrics["round_work_seconds"] <= 0:
            raise ValueError("Invalid round duration")
        client_metrics = []
        for client in sorted(clients):
            if set(clients[client]) != expected_rounds:
                raise ValueError("Missing client round: " + client)
            cm = clients[client][index]
            expected = partition["clients"][client]["train"]["samples"] * settings["local_epochs"]
            if cm["processed_samples"] != expected or cm["gpu_enabled"] != 0:
                raise ValueError("Processed sample count or CPU policy differs: " + client)
            client_metrics.append(cm)
        processed = sum(c["processed_samples"] for c in client_metrics)
        payload = sum(c["global_checkpoint_bytes"] + c["update_bytes"] for c in client_metrics)
        duration = metrics["round_work_seconds"]
        records.append(
            {
                "round": index,
                "clients": len(clients),
                "duration_seconds": duration,
                "training_examples": processed,
                "training_examples_per_second": processed / duration,
                "accepted_updates_per_second": len(clients) / duration,
                "client_model_payload_bytes": payload,
                "client_model_payload_bytes_per_second": payload / duration,
                "global_test_accuracy": metrics["global_test_accuracy"],
                "global_test_macro_f1": metrics["global_test_macro_f1"],
            }
        )
    resource = resource_summary(
        samples, ["ws-server", "vessel-coordinator", *(f"vessel-{i + 1}" for i in range(len(clients)))]
    )
    for role, values in resource["roles"].items():
        config = settings["resources"][
            "hub" if role == "ws-server" else ("coordinator" if role == "vessel-coordinator" else "client")
        ]
        values["cpu_limit_cores"] = quantities(config["limits"]["cpu"])
        values["memory_limit_mib"] = quantities(config["limits"]["memory"], True)
        values["mean_cpu_percent_of_limit"] = 100 * values["mean_cpu_cores"] / values["cpu_limit_cores"]
        values["peak_memory_percent_of_limit"] = 100 * values["peak_memory_mib"] / values["memory_limit_mib"]
    return {
        "rounds": records,
        "resources": resource,
        "mean_round_seconds": statistics.mean(r["duration_seconds"] for r in records),
        "mean_examples_per_second": sum(r["training_examples"] for r in records)
        / sum(r["duration_seconds"] for r in records),
        "mean_updates_per_second": len(clients) * len(records) / sum(r["duration_seconds"] for r in records),
        "mean_client_payload_bytes_per_second": sum(r["client_model_payload_bytes"] for r in records)
        / sum(r["duration_seconds"] for r in records),
    }


def write_csv(path, records):
    """Write portable tabular measurements without pandas."""
    if not records:
        path.unlink(missing_ok=True)
        return
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def curves(destination, results):
    """Export every repetition and a standalone SVG of measured points, without extrapolation."""
    designs = {
        report.get("study_design", "fixed_total") for key, report in results.items() if not key.startswith("smoke-")
    }
    if len(designs) > 1:
        raise ValueError("Scaling study designs must have separate curves")
    design = next(iter(designs), "fixed_total")
    records = []
    for key, report in results.items():
        if key.startswith("smoke-"):
            continue
        summary = report["summary"]
        records.append(
            {
                "job": key,
                "study_design": design,
                "clients": report["clients"],
                "repetition": report["repetition"],
                "rounds": len(summary["rounds"]),
                "processed_training_examples_per_round": summary["rounds"][0]["training_examples"],
                "mean_round_seconds": summary["mean_round_seconds"],
                "training_examples_per_second": summary["mean_examples_per_second"],
                "accepted_updates_per_second": summary["mean_updates_per_second"],
                "client_payload_bytes_per_second": summary["mean_client_payload_bytes_per_second"],
                "aggregate_mean_cpu_cores": summary["resources"]["aggregate_mean_cpu_cores"],
                "aggregate_mean_memory_mib": summary["resources"]["aggregate_mean_memory_mib"],
            }
        )
    records.sort(key=lambda r: (r["clients"], r["repetition"]))
    write_csv(destination / "scaling.csv", records)
    if not records:
        (destination / "scaling.svg").unlink(missing_ok=True)
        return
    panels = [
        ("mean_round_seconds", "Round latency (s)"),
        ("training_examples_per_second", "Training examples / s"),
        ("aggregate_mean_cpu_cores", "Aggregate CPU cores"),
        ("aggregate_mean_memory_mib", "Aggregate memory (MiB)"),
    ]
    label = "fixed workload per client; reused samples" if design == "fixed_per_client" else "fixed total data"
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="680" viewBox="0 0 1000 680">',
        '<rect width="1000" height="680" fill="white"/>',
        '<g font-family="sans-serif" font-size="13" fill="#18283b">',
        f'<text x="40" y="25">Single-host FL scaling — {label}, CPU</text>',
    ]
    for index, (field, title) in enumerate(panels):
        x, y = 65 + (index % 2) * 490, 75 + (index // 2) * 295
        maximum = max(r[field] for r in records) * 1.1 or 1
        counts = sorted({r["clients"] for r in records})
        points = []
        svg.extend(
            [
                f'<text x="{x}" y="{y - 15}">{title}</text>',
                f'<path d="M{x},{y} v210 h380" fill="none" stroke="#64748b"/>',
            ]
        )
        for count in counts:
            value = statistics.mean(r[field] for r in records if r["clients"] == count)
            px, py = x + (count - 1) * 380 / max(1, max(counts) - 1), y + 210 - value / maximum * 200
            points.append(f"{px},{py}")
            svg.extend(
                [
                    f'<circle cx="{px}" cy="{py}" r="4" fill="#1765b5"/>',
                    f'<text x="{px - 8}" y="{py - 10}">{value:.3g}</text>',
                    f'<text x="{px - 5}" y="{y + 230}">{count}</text>',
                ]
            )
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#1765b5"/>')
        svg.append(f'<text x="{x + 130}" y="{y + 255}">Logical FL clients</text>')
    svg.append("</g></svg>")
    (destination / "scaling.svg").write_text("".join(svg))
