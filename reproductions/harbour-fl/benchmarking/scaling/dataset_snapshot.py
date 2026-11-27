"""Copy prepared features into a private workload volume and verify the copied bytes."""

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath


def copy_verified(source, destination, hashes):
    """Keep live preparation changes out of training after a verified snapshot has been sealed."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    destination.chmod(0o755)
    staging = destination / ".snapshot-staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(mode=0o700)
    for name, expected in hashes.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("Feature path must be dataset-relative: " + name)
        original = (source / name).resolve()
        if not original.is_relative_to(source):
            raise ValueError("Feature path escapes the prepared dataset: " + name)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            with target.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() == expected:
                    continue
        descriptor, temporary_name = tempfile.mkstemp(dir=staging, prefix=".feature-")
        temporary = Path(temporary_name)
        checksum = hashlib.sha256()
        try:
            with (
                os.fdopen(descriptor, "wb") as outgoing,
                original.open("rb") as incoming,
            ):
                while block := incoming.read(1024 * 1024):
                    checksum.update(block)
                    outgoing.write(block)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            if checksum.hexdigest() != expected:
                raise ValueError("Prepared feature changed before snapshot: " + name)
            temporary.chmod(0o444)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    staging.rmdir()
    for folder in sorted((p for p in destination.rglob("*") if p.is_dir()), reverse=True):
        folder.chmod(0o555)
    destination.chmod(0o555)


if __name__ == "__main__":
    copy_verified(sys.argv[1], sys.argv[2], json.loads(Path(sys.argv[3]).read_text()))
