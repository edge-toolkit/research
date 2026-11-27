"""Bounded local commands and content-addressed evidence for the scaling study."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))

from study_support import controller_source


def support_source(root=ROOT):
    """Resolve the shared study helper in a checkout or portable package."""
    return root / "study_support.py" if (root / "study_support.py").is_file() else root.parent / "study_support.py"


def load(path):
    """Read JSON without depending on the application environment."""
    return json.loads(Path(path).read_text())


def digest(value):
    """Hash a JSON value deterministically."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    """Hash the bytes of an artifact."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources(root=ROOT):
    """Use the same explicit source manifest for packaging and experiment identity."""
    return [
        root / name
        for name in (
            ".gitignore",
            "README.md",
            "capacity.py",
            "cli.py",
            "cluster.py",
            "common.py",
            "dataset_snapshot.py",
            "measurements.py",
            "mise.lock",
            "mise.toml",
            "package.py",
            "partitions.py",
            "pyproject.toml",
            "runner.py",
            "settings-per-client.json",
            "settings.json",
            "test_extension.py",
            "test_per_client.py",
            "test_safety.py",
            "test_scaling.py",
            "uv.lock",
        )
        if (root / name).is_file() and not (root / name).is_symlink()
    ]


def tool_identity(root=ROOT):
    """Invalidate resumption whenever source, documentation or default settings change."""
    files = {str(p.relative_to(root)): sha(p) for p in sources(root) if "controller" not in p.relative_to(root).parts}
    files["study_support.py"] = sha(support_source(root))
    source = controller_source(root)
    return digest(
        {
            "suite": files,
            "controller": {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob("*.py"))},
        }
    )


def save(path, value):
    """Flush an atomic JSON replacement so interrupted runs retain their journal."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class CommandError(RuntimeError):
    """Retain command status so authorization denial is distinguishable from transport failure."""

    def __init__(self, argv, result):
        super().__init__(f"{argv[0]} exited {result.returncode}: {result.stderr[-2500:]}")
        self.returncode, self.stdout, self.stderr = result.returncode, result.stdout, result.stderr


def command(argv, timeout=60, stdin=None):
    """Run without a shell and surface bounded stderr when a subprocess fails."""
    result = subprocess.run(
        [str(x) for x in argv], input=stdin, capture_output=True, text=True, timeout=timeout, check=False
    )
    if result.returncode:
        raise CommandError(argv, result)
    return result.stdout
