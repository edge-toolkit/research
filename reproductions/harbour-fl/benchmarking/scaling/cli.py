"""Select the existing host mise/Python environment before running cluster operations."""

import argparse
import os
import shutil
import sys
from pathlib import Path

from common import ROOT, load


def parser():
    """Keep launcher and runner command options identical."""
    result = argparse.ArgumentParser(description="Run the Harbour-FL participant scaling study", allow_abbrev=False)
    result.add_argument("action", choices=("plan", "check", "smoke", "run", "status", "cleanup"))
    result.add_argument("--settings", type=Path, default=ROOT / "settings-per-client.json")
    result.add_argument("--output", type=Path)
    result.add_argument("--case", type=int, choices=(1, 3, 6, 12, 24))
    return result


def arguments(values):
    """Resolve operator paths before mise -C changes the working directory."""
    result = list(values)
    for i, token in enumerate(result):
        key, sep, value = token.partition("=")
        if key not in ("--settings", "--output"):
            continue
        if not sep:
            if i + 1 == len(result) or result[i + 1].startswith("--"):
                raise ValueError(key + " requires a path")
            result[i + 1] = str(Path(result[i + 1]).expanduser().resolve())
        else:
            if not value:
                raise ValueError(key + " requires a path")
            result[i] = key + "=" + str(Path(value).expanduser().resolve())
    return result


def main():
    """Launch benchmark commands in the configured Harbour-FL environment."""
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        parser().print_help()
        return
    values = arguments(sys.argv[1:])
    args = parser().parse_args(values)
    if args.action == "smoke" and args.case not in (None, 1):
        raise ValueError("Smoke supports only --case 1")
    runner = [str(ROOT / "runner.py"), *values]
    if args.action in ("plan", "status"):
        os.execv(sys.executable, [sys.executable, *runner])
    from cluster import locate

    app = locate(load(args.settings))
    mise = shutil.which("mise") or str(Path.home() / ".local/bin/mise")
    if not Path(mise).is_file():
        raise RuntimeError("Activate the host mise installation first")
    os.execv(mise, [mise, "-C", str(app), "exec", "--", str(app / ".venv/bin/python"), *runner])


if __name__ == "__main__":
    main()
