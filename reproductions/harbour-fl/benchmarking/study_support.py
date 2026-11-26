"""Resolve maintained Harbour-FL code and its shared controller for deployment studies."""

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path


def application(settings, root):
    """Use an explicit application directory, or the enclosing Harbour-FL checkout."""
    app = Path(settings["application"]).expanduser() if settings.get("application") else root.parents[1]
    app = (app if app.is_absolute() else root / app).resolve()
    required = ("src/harbour_fl/workload.py", "benchmarking/integration.py", ".venv/bin/python", "recovery.json")
    if any(not (app / name).is_file() for name in required):
        raise RuntimeError("Set application to a prepared Harbour-FL directory; run mise run setup:benchmark there")
    return app


def controller_source(root):
    """Find canonical controller source in the checkout or a portable study package."""
    candidates = [root / "controller/src"]
    candidates.extend(parent / "experiments/reliability-benchmarks/controllers/src" for parent in root.parents)
    for candidate in candidates:
        if (candidate / "et_experiments/benchmarking/engine.py").is_file():
            return candidate
    raise RuntimeError("Shared benchmark controller is missing; use a checkout or a complete source package")


def source_identity(app):
    """Bind receipts to application and adapter source."""
    files = [*(app / "src/harbour_fl").rglob("*.py"), *(app / "benchmarking").glob("*.py"), app / "recovery.json"]
    return {
        str(path.relative_to(app)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
        if path.is_file()
    }


def adapter(app, root, settings, workers=None):
    """Load the application's current adapter with an explicit study context and namespace."""
    os.environ["HARBOUR_FL_KUBE_CONTEXT"] = settings["context"]
    sys.path[:0] = [str(controller_source(root)), str(app / "benchmarking"), str(app / "src")]
    path = app / "benchmarking/integration.py"
    name = "harbour_study_adapter_" + hashlib.sha256(str(app).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.NAMESPACE = settings.get("namespace", settings.get("source_namespace", "et-harbour-fl"))
    instance = module.HarbourFLAdapter()
    instance.config = replace(
        instance.config,
        context=settings["context"],
        namespace=module.NAMESPACE,
        workers=tuple(workers) if workers is not None else instance.config.workers,
    )
    return module, instance


def kubectl_prefix(settings):
    """Preserve the operator's kubeconfig selection without changing global context."""
    result = ["kubectl"]
    if settings.get("kubeconfig"):
        result.extend(("--kubeconfig", str(Path(settings["kubeconfig"]).expanduser())))
    return [*result, "--context", settings["context"]]


def prepared_metadata(info, data):
    """Bind generated partition manifests to unchanged, read-only pretrained feature archives."""
    value = {key: item for key, item in info.items() if key != "fingerprint"}
    value["files"] = {
        **info["files"],
        **{"/scaling-source/" + name: checksum for name, checksum in info["files"].items()},
        **{name: hashlib.sha256(text.encode()).hexdigest() for name, text in data.items()},
    }
    value["fingerprint"] = hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return json.dumps(value, indent=2, sort_keys=True) + "\n"
