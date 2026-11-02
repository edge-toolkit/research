"""Download or import pinned pretrained artifacts, verifying their recorded SHA-256 identities."""

import argparse
import json
import os
import shutil
import urllib.request
from pathlib import Path

from .provenance import digest, file_hash, save

MANIFEST = Path(__file__).with_suffix(".json")


def identity():
    """Return the source and weight identities included with the application, without loading any weights."""
    return json.loads(MANIFEST.read_text())


def assets_root():
    return Path(
        os.environ.get("HARBOUR_FL_PRETRAINED_ASSETS", Path(__file__).resolve().parents[2] / "models/pretrained")
    )


def verify(root):
    """Require every pinned model and encoder-source artifact to exist with its recorded SHA-256 digest."""
    expected = identity()
    for name, item in expected["files"].items():
        path = Path(root) / name
        if not path.is_file() or file_hash(path) != item["sha256"]:
            raise ValueError(f"missing or changed pretrained artifact: {name}; run mise run setup:weights")
    return expected


def prepare_assets(root, source=None):
    """Keep weights outside Git; accept an existing verified artifact directory or fetch upstream originals."""
    root = Path(root)
    expected = identity()
    for name, item in expected["files"].items():
        target = root / name
        if target.is_file() and file_hash(target) == item["sha256"]:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".download")
        if source is not None:
            shutil.copyfile(Path(source) / name, temporary)
        elif "url" in item or "upstream_path" in item:
            print(f"Downloading {name}", flush=True)
            url = item.get("url") or (
                expected["efficientat_repository"] + expected["efficientat_revision"] + "/" + item["upstream_path"]
            )
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            if name.startswith("efficientat/"):
                if file_hash(temporary) != item["upstream_sha256"]:
                    raise ValueError(f"upstream source hash mismatch: {name}")
                content = temporary.read_text().replace("from models.mn.", "from .")
                content = content.replace("from helpers.utils import NAME_TO_WIDTH", "from .width import NAME_TO_WIDTH")
                temporary.write_text(content)
        else:
            content = (
                ""
                if name.endswith("__init__.py")
                else (
                    'def NAME_TO_WIDTH(name):\n    if not name.startswith("mn10"):\n'
                    '        raise ValueError("Only MN10 is supported")\n    return 1.0\n'
                )
            )
            temporary.write_text(content)
        if file_hash(temporary) != item["sha256"]:
            temporary.unlink()
            raise ValueError(f"pretrained artifact hash mismatch: {name}")
        temporary.replace(target)
    verify(root)
    save(root / "assets.json", expected)
    return expected


def fingerprint():
    return digest(identity())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=assets_root())
    parser.add_argument("--source", type=Path, help="Import an existing artifact directory instead of downloading")
    args = parser.parse_args()
    prepare_assets(args.output, args.source)
    print(f"Verified pretrained artifacts: {args.output}")


if __name__ == "__main__":
    main()
