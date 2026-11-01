#!/usr/bin/env python3
"""Apply project-specific settings to generated Edge Toolkit deployments."""

import argparse
from pathlib import Path

from deployment_support.credentials import retain_credentials
from deployment_support.k3s import configure_k3s
from deployment_support.release import prepare_release
from deployment_support.scenario import configure_mise, configure_scenario


def main() -> int:
    """Dispatch the selected local-release, credential or deployment adaptation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("mise", "k3s", "scenario", "core-release", "credentials"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.kind == "mise":
        configure_mise(args.path)
    elif args.kind == "k3s":
        configure_k3s(args.path)
    elif args.kind == "scenario":
        configure_scenario(args.path)
    elif args.kind == "credentials":
        retain_credentials(args.path)
    else:
        prepare_release(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
