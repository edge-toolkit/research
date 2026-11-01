"""Harbour-FL deployment support: credentials."""

from __future__ import annotations

import os
from pathlib import Path


def retain_credentials(generated: Path) -> None:
    """Reuse project-local credentials when regenerating deployments backed by persistent service databases."""
    stable = generated.parent.parent / "deployment-secrets" / "secrets.env"
    stable.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    content = stable.read_bytes() if stable.exists() else generated.read_bytes()
    values = dict(
        line.split("=", 1)
        for line in content.decode().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    passwords = [values.get(name) for name in ("ZO_ROOT_USER_PASSWORD", "OTLP_AUTH_PASSWORD")]
    if not all(passwords) or passwords[0] != passwords[1]:
        raise ValueError("stored OpenObserve and OTLP passwords must be nonempty and match")
    if not stable.exists():
        with stable.open("xb") as destination:
            os.chmod(stable, 0o600)
            destination.write(content)
    os.chmod(stable, 0o600)
    generated.write_bytes(content)
    os.chmod(generated, 0o600)
