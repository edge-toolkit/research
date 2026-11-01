"""Map Croatia paired samples into new recording-disjoint, content-disjoint study splits."""

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .provenance import digest, file_hash, save

LABEL_MAP = {
    "fishing boats": "fishing",
    "fishing boat": "fishing",
    "fishing": "fishing",
    "ferries": "passenger",
    "ferry": "passenger",
    "tour boats": "passenger",
    "tour boat": "passenger",
    "tour boat j": "passenger",
    "tour boat f": "passenger",
    "passenger": "passenger",
    "motor boats": "pleasurecraft",
    "motorboats": "pleasurecraft",
    "yachts": "pleasurecraft",
    "yacht": "pleasurecraft",
    "pleasurecraft": "pleasurecraft",
    "sail boats": "sailing",
    "sail boat (with motor)": "sailing",
    "sailing": "sailing",
    "aux vessels": "workboat",
    "aux vessel": "workboat",
    "workboat": "workboat",
}


def read_csv(path):
    """Stream the combined manifest without materializing unrelated ONC rows."""
    with Path(path).open(newline="") as source:
        yield from csv.DictReader(source)


def relocated_paths(value, raw_root):
    """Keep the complete class-relative path, including nested ferry/tour-boat vessel folders."""
    path = Path(value)
    if not raw_root:
        return [path]
    matches = [
        Path(raw_root) / Path(*path.parts[index:])
        for index, part in enumerate(path.parts[:-1])
        if part.strip().casefold() in LABEL_MAP
    ]
    return matches or [Path(raw_root) / Path(*path.parts[-3:])]


def unique_existing(candidates):
    """Resolve existing candidate paths, rejecting mappings that point to more than one distinct asset."""
    existing = {p.resolve() for p in candidates if p.is_file()}
    if len(existing) > 1:
        raise ValueError("ambiguous raw asset mapping; review the source path and raw-root")
    return next(iter(existing), None)


def resolve_audio(row, raw_root):
    """Locate the manifest recording under the selected raw-data root or report the unavailable WAV."""
    candidates = [
        candidate
        for key in ("resolved_audio_path", "path")
        if row.get(key)
        for candidate in relocated_paths(row[key], raw_root)
    ]
    found = unique_existing(candidates)
    if found:
        return found
    raise ValueError(f"raw WAV unavailable for {row['segment_id']}; provide --raw-root")


def resolve_image(row, dataset, raw_root):
    """Use the manifest's exact pairing, including older manifests with no copied-image column."""
    candidates = []
    prepared = row.get("camera_image_output_path", "")
    original = row.get("camera_image_path", "")
    if prepared:
        copied = Path(dataset) / prepared
        if copied.is_file():
            return copied.resolve()
    if original:
        path = Path(original)
        candidates.extend(relocated_paths(path, raw_root) if raw_root or path.is_absolute() else [Path(dataset) / path])
    return unique_existing(candidates)


def prepare(dataset, output, *, raw_root=None, seed=17):
    """Filter Croatia pairs, group duplicate content and create class-covered, client-covered disjoint partitions."""
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    source = dataset / "segment_manifest.csv"
    identity = {
        "manifest_sha256": file_hash(source),
        "dataset": str(dataset),
        "raw_root": str(Path(raw_root).resolve()) if raw_root else None,
        "seed": seed,
        "schema": 3,
        "source_filter": "Croatia",
    }
    if (output / "dataset.json").exists():
        old = json.loads((output / "dataset.json").read_text())
        if old["input"] != identity:
            raise ValueError("prepared inputs changed; select a new --prepared directory")
        load_dataset(output, verify_files=True)
        validate_client_coverage(old["rows"])
        return old
    if output.exists() and any(output.iterdir()):
        raise ValueError("prepared directory is nonempty without a completed manifest")
    rows, excluded, hashes, ids, paired_contents = [], [], {}, {}, {}
    source_counts = Counter()
    manifest_rows = selected_source_rows = inferred_source_rows = 0
    print(f"Scanning {source}; selecting Croatia records only", flush=True)
    for row in read_csv(source):
        manifest_rows += 1
        # A source-less legacy manifest is accepted; mixed manifests must identify Croatia explicitly.
        origin = row.get("source_dataset", "").strip()
        if "source_dataset" not in row:
            origin = "Croatia"
            inferred_source_rows += 1
        source_counts[origin or "unknown"] += 1
        if manifest_rows % 100000 == 0:
            print(f"Scanned {manifest_rows:,} manifest rows; {selected_source_rows:,} Croatia rows", flush=True)
        if origin.casefold() != "croatia":
            continue
        selected_source_rows += 1
        sid = row["segment_id"]
        if row.get("is_usable", "true").lower() not in {"true", "1"}:
            excluded.append({"segment_id": sid, "reason": "source marks audio unusable"})
            continue
        original = row.get("source_label") or row["label"]
        label = LABEL_MAP.get(original.strip().lower())
        if not label or (row.get("standard_label") and label != row["standard_label"]):
            raise ValueError(f"unmapped or inconsistent label {original!r}: review LABEL_MAP")
        image = resolve_image(row, dataset, raw_root)
        if image is None:
            excluded.append({"segment_id": sid, "reason": "paired camera image unavailable"})
            continue
        audio = resolve_audio(row, raw_root)
        event = row.get("event_id")
        if not event:
            raise ValueError(f"missing recording/event identifier for {sid}")
        for path in (audio, image):
            key = str(path.resolve())
            if key not in hashes:
                hashes[key] = file_hash(path)
        start, end = float(row["segment_start_sec"]), float(row["segment_end_sec"])
        if start < 0 or end <= start or end - start > 30:
            raise ValueError(f"invalid segment bounds for {sid}")
        # Old segment IDs contain only the WAV basename; qualify by the recording's class/event path.
        recording = (
            audio.relative_to(Path(raw_root).resolve()).as_posix() if raw_root else Path(*audio.parts[-3:]).as_posix()
        )
        qualified_id = sid + "__" + digest({"recording": recording})[:12]
        content = (hashes[str(audio)], hashes[str(image)], start, end)
        signature = (label, *content)
        if qualified_id in ids and ids[qualified_id] != signature:
            raise ValueError(f"conflicting segment identity: {sid} in {recording}")
        ids[qualified_id] = signature
        if content in paired_contents:
            kept_id, kept_label = paired_contents[content]
            if label != kept_label:
                raise ValueError(f"identical paired segment has conflicting labels: {sid}")
            excluded.append(
                {"segment_id": sid, "reason": "duplicate paired segment content", "kept_segment_id": kept_id}
            )
            continue
        paired_contents[content] = (qualified_id, label)
        rows.append(
            {
                "segment_id": qualified_id,
                "source_segment_id": sid,
                "source_dataset": "Croatia",
                "source_label": original,
                "label": label,
                "event_id": event,
                "audio": str(audio),
                "image": str(image.resolve()),
                "start": start,
                "end": end,
                "audio_sha256": hashes[str(audio)],
                "image_sha256": hashes[str(image.resolve())],
            }
        )
    if not selected_source_rows:
        raise ValueError("no Croatia records found in the source manifest")
    if not rows:
        raise ValueError("no usable paired Croatia samples; check the raw-root and camera image paths")
    parent = {r["event_id"]: r["event_id"] for r in rows}

    def find(key):
        """Find and compress a recording group representative while merging shared-content events."""
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    seen = {}
    for row in rows:
        for key in ("audio_sha256", "image_sha256"):
            token = (key, row[key])
            if token in seen:
                left, right = sorted((find(row["event_id"]), find(seen[token])))
                parent[right] = left
            seen[token] = row["event_id"]
    groups, by_label = defaultdict(list), defaultdict(list)
    for row in rows:
        row["group"] = find(row["event_id"])
        groups[row["group"]].append(row)
    for group, members in groups.items():
        labels = {r["label"] for r in members}
        if len(labels) != 1:
            raise ValueError(f"shared recording/image has conflicting labels in group {group}")
        by_label[next(iter(labels))].append(group)
    included = []
    for label, names in sorted(by_label.items()):
        if len(names) < 3:
            excluded.extend(
                {"segment_id": r["segment_id"], "reason": "fewer than three independent groups", "label": label}
                for name in names
                for r in groups[name]
            )
            continue
        random.Random(f"{seed}:{label}").shuffle(names := sorted(names))
        n_test = n_val = max(1, round(len(names) * 0.15))
        for index, name in enumerate(names):
            split = "test" if index < n_test else "val" if index < n_test + n_val else "train"
            client = f"client-{int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 3 + 1}"
            included.extend({**r, "split": split, "client": client} for r in groups[name])
    labels = sorted({r["label"] for r in included})
    if len(labels) < 2:
        raise ValueError("at least two classes with three independent groups are required")
    included.sort(key=lambda r: r["segment_id"])
    for split in ("train", "val", "test"):
        if {r["label"] for r in included if r["split"] == split} != set(labels):
            raise ValueError("each split must contain all study classes")
    validate_client_coverage(included)
    document = {
        "input": identity,
        "labels": labels,
        "label_to_index": {v: i for i, v in enumerate(labels)},
        "source_label_map": LABEL_MAP,
        "rows": included,
        "excluded": excluded,
        "split_counts": {
            s: dict(Counter(r["label"] for r in included if r["split"] == s)) for s in ("train", "val", "test")
        },
        "independent_groups_per_class": {k: len(v) for k, v in by_label.items()},
        "scope": "recording/event-disjoint; not verified vessel-identity or date-disjoint",
        "source_files": hashes,
        "import_summary": {
            "manifest_rows": manifest_rows,
            "selected_source_rows": selected_source_rows,
            "skipped_other_source_rows": manifest_rows - selected_source_rows,
            "source_counts": dict(source_counts),
            "source_inferred_for_legacy_rows": inferred_source_rows,
            "paired_rows_before_rare_class_exclusion": len(rows),
            "included_rows": len(included),
            "excluded_selected_source_rows": len(excluded),
            "duplicate_paired_segments": sum(r["reason"] == "duplicate paired segment content" for r in excluded),
        },
    }
    document["fingerprint"] = digest(document)
    save(output / "dataset.json", document)
    for split in ("train", "val", "test"):
        selected = [r for r in included if r["split"] == split]
        with (output / f"{split}.csv").open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
    save(output / "excluded.json", excluded)
    return document


def validate_client_coverage(rows):
    """Require training samples for each logical client before expensive feature extraction begins."""
    for client in ("client-1", "client-2", "client-3"):
        if not any(row["split"] == "train" and row["client"] == client for row in rows):
            raise ValueError(f"{client} has no training samples; use more recording groups or another split seed")


def load_dataset(prepared, *, verify_files=False):
    """Verify the saved dataset fingerprint and optionally recheck every referenced source-file hash."""
    value = json.loads((Path(prepared) / "dataset.json").read_text())
    if digest({k: v for k, v in value.items() if k != "fingerprint"}) != value["fingerprint"]:
        raise ValueError("prepared dataset metadata changed")
    if verify_files:
        for path, expected in value["source_files"].items():
            if not Path(path).is_file() or file_hash(path) != expected:
                raise ValueError(f"source asset changed: {path}")
    return value
