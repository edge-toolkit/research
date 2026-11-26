"""Fixed-total-data, approximately stratified partitions of the original prepared sample sets."""

import collections
import csv
import io
import random
from pathlib import PurePosixPath

from common import digest
from study_support import prepared_metadata


def rows(text):
    """Decode a prepared CSV manifest."""
    return list(csv.DictReader(io.StringIO(text)))


def csv_text(values, fields):
    """Emit only fields needed by the recorded application loader."""
    out = io.StringIO()
    writer = csv.DictWriter(out, fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(values)
    return out.getvalue()


def union(manifests):
    """Deduplicate shared holdouts while rejecting conflicting records for one segment."""
    records = {}
    for text in manifests:
        for row in rows(text):
            value = {key: row[key] for key in ("segment_id", "standard_label", "feature_path")}
            previous = records.setdefault(value["segment_id"], value)
            if previous != value:
                raise ValueError("Conflicting sample: " + value["segment_id"])
    return [records[key] for key in sorted(records)]


def balance(records, count, seed):
    """Distribute each class to the least populated shard; total sizes differ by at most one."""
    if len(records) < count:
        raise ValueError("Not enough samples for nonempty client partitions")
    groups = collections.defaultdict(list)
    for row in records:
        groups[row["standard_label"]].append(row)
    result = [[] for _ in range(count)]
    rng = random.Random(seed)
    for label in sorted(groups):
        group = sorted(groups[label], key=lambda r: r["segment_id"])
        rng.shuffle(group)
        # Re-sort by current size before each pass, rather than restarting every class at client 1.
        for start in range(0, len(group), count):
            order = sorted(range(count), key=lambda i: (len(result[i]), i))
            for position, row in enumerate(group[start : start + count]):
                result[order[position]].append(row)
    if max(map(len, result)) - min(map(len, result)) > 1:
        raise RuntimeError("Partition sizes differ by more than one sample")
    return result


def asset_path(value):
    """Keep every asset under the read-only prepared dataset mount."""
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError("Expected a dataset-relative asset path: " + value)
    return "/scaling-source/" + str(path)


def _prepare_total(snapshot, count, seed):
    """Retain the original train/validation/test sets and change only client assignments."""
    fields = ("segment_id", "standard_label", "feature_path")
    manifests = dict(snapshot["manifests"])
    for name, ids in snapshot["usable_ids"].items():
        selected = set(ids)
        manifests[name] = csv_text([row for row in rows(manifests[name]) if row["segment_id"] in selected], fields)
    train = union([text for name, text in manifests.items() if name.endswith("/train_manifest.csv")])
    val = union([text for name, text in manifests.items() if name.endswith("/val_manifest.csv")])
    test = union([manifests["test_manifest.csv"]])
    ids = [{r["segment_id"] for r in split} for split in (train, val, test)]
    if any(ids[a] & ids[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Segment leakage between original splits; review before benchmarking")
    features = {r["segment_id"]: r["feature_path"] for r in rows(manifests["feature_manifest.csv"])}
    required = ids[0] | ids[1] | ids[2]
    if not required.issubset(features):
        raise ValueError("Missing prepared features")
    converted = {}
    for split in (train, val, test):
        for row in split:
            converted[row["segment_id"]] = {
                **row,
                "feature_path": asset_path(row["feature_path"]),
            }
    data = {
        "class_summary.csv": manifests["class_summary.csv"],
        "feature_manifest.csv": csv_text(
            [{"segment_id": key, "feature_path": asset_path(features[key])} for key in sorted(required)],
            ("segment_id", "feature_path"),
        ),
        "test_manifest.csv": csv_text([converted[r["segment_id"]] for r in test], fields),
    }
    split_names = ("train", "validation", "test")
    report = {
        "count": count,
        "seed": seed,
        "samples": {name: len(ids[i]) for i, name in enumerate(split_names)},
        "split_hashes": {name: digest(sorted(ids[i])) for i, name in enumerate(split_names)},
        "partition_unit": "prepared segment; original recording-level split limitations remain",
        "original_manifest_coverage": snapshot.get("manifest_counts", {}),
        "clients": {},
    }
    train_parts, val_parts = balance(train, count, seed), balance(val, count, seed + 1)
    for i in range(count):
        client = f"vessel_{i + 1}"
        report["clients"][client] = {}
        for split, values in (("train", train_parts[i]), ("val", val_parts[i])):
            data[f"clients/{client}/{split}_manifest.csv"] = csv_text(
                [converted[r["segment_id"]] for r in values], fields
            )
            report["clients"][client][split] = {
                "samples": len(values),
                "classes": dict(collections.Counter(r["standard_label"] for r in values)),
                "ids": [r["segment_id"] for r in values],
            }
    return data, report


def repeated_shards(records, count, size, seed):
    """Append stable client workloads from a cyclic, approximately stratified ordering of the source pool."""
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for row in records:
        groups[row["standard_label"]].append(row)
    ordered = []
    for label in sorted(groups):
        values = sorted(groups[label], key=lambda r: r["segment_id"])
        rng.shuffle(values)
        ordered.extend(((i + 0.5) / len(values), label, row) for i, row in enumerate(values))
    pool = [item[2] for item in sorted(ordered, key=lambda item: (item[0], item[1]))]
    if not pool or type(size) is not int or size <= 0:
        raise ValueError("Fixed per-client workloads require a nonempty pool and positive sample count")
    return [[pool[(client * size + i) % len(pool)] for i in range(size)] for client in range(count)]


def prepare(snapshot, count, seed, study=None):
    """Keep fixed-total partitions unchanged, or simulate a fixed workload per client with disclosed sample reuse."""
    study = study or {}
    design = study.get("study_design", "fixed_total")
    if design == "fixed_total":
        data, report = _prepare_total(snapshot, count, seed)
        data["preparation.json"] = prepared_metadata(snapshot["preparation"], data)
        return data, report
    if design != "fixed_per_client":
        raise ValueError("Unknown scaling study design")
    # One base shard supplies each complete source pool after the original loader exclusions and split checks.
    data, report = _prepare_total(snapshot, 1, seed)
    pools = {split: rows(data.pop(f"clients/vessel_1/{split}_manifest.csv")) for split in ("train", "val")}
    report.update(
        count=count,
        study_design=design,
        source_split_hashes=report["split_hashes"],
        pool_samples=dict(report["samples"]),
        clients={},
        reuse={},
    )
    for split, name, size in (
        ("train", "train", study["train_samples_per_client"]),
        ("val", "validation", study["validation_samples_per_client"]),
    ):
        shards = repeated_shards(pools[split], count, size, seed + (split == "val"))
        usage = collections.Counter(row["segment_id"] for shard in shards for row in shard)
        report["samples"][name] = count * size
        report["reuse"][name] = {
            "assigned_occurrences": count * size,
            "unique_samples": len(usage),
            "reused_occurrences": count * size - len(usage),
            "maximum_uses_of_one_sample": max(usage.values()),
        }
        report["split_hashes"] = {**report["split_hashes"], name: digest(sorted(usage))}
        for i, shard in enumerate(shards):
            client = f"vessel_{i + 1}"
            data[f"clients/{client}/{split}_manifest.csv"] = csv_text(
                shard, ("segment_id", "standard_label", "feature_path")
            )
            report["clients"].setdefault(client, {})[split] = {
                "samples": len(shard),
                "unique_samples": len({r["segment_id"] for r in shard}),
                "classes": dict(collections.Counter(r["standard_label"] for r in shard)),
                "ids": [r["segment_id"] for r in shard],
            }
    report["note"] = "Prepared sample reuse simulates resource demand; it does not create independent vessel data."
    data["preparation.json"] = prepared_metadata(snapshot["preparation"], data)
    return data, report
