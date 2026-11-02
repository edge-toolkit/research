"""Research adds domain settings but leaves recovery resources to Core scenario generation."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tomllib
import yaml
from deployment_support import detector_environment
from deployment_support.credentials import retain_credentials
from deployment_support.k3s import REGISTRY_INITIALIZER, configure_k3s
from deployment_support.release import prepare_release
from deployment_support.scenario import configure_mise, configure_scenario

PROJECT = Path(__file__).resolve().parents[1]


class DeploymentTests(unittest.TestCase):
    def test_mlflow_task_creates_results_before_starting_the_server(self):
        """Start from an empty directory and retain the configured database and artifact paths."""
        task = tomllib.loads((PROJECT / "mise.toml").read_text())["tasks"]["mlflow"]["run"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interpreter = root / ".venv/bin/python"
            interpreter.parent.mkdir(parents=True)
            interpreter.write_text("#!/bin/sh\n[ -d results ] || exit 9\nprintf '%s\\n' \"$@\"\n")
            interpreter.chmod(0o755)
            result = subprocess.run(["/bin/sh", "-e", "-c", task], cwd=root, check=True, text=True, capture_output=True)
            self.assertTrue((root / "results").is_dir())
            self.assertIn("sqlite:///results/mlflow.db", result.stdout.splitlines())
            self.assertIn("results/mlflow-artifacts", result.stdout.splitlines())

    def test_native_deployment_uses_the_active_interpreters_package_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mise.toml"
            path.write_text('[env]\nMODULES_ROOT = ""\nPYO3_PYTHONPATH = "../src:/opt/et-python"\n')
            with patch(
                "deployment_support.scenario.site.getsitepackages", return_value=["/venv/lib/pythonX/site-packages"]
            ):
                configure_mise(path)
            environment = tomllib.loads(path.read_text())["env"]
            self.assertIn("/venv/lib/pythonX/site-packages", environment["PYO3_PYTHONPATH"])
            self.assertNotIn("python3.13", environment["PYO3_PYTHONPATH"])

    def test_detection_deployment_is_opt_in_and_uses_bundled_model_paths(self):
        with patch.dict(os.environ, {"HARBOUR_FL_DETECTOR_ENABLED": "false"}):
            self.assertEqual(detector_environment(container=True)["HARBOUR_FL_DETECTOR_ENABLED"], "false")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models/vessel-detector"
            models.mkdir(parents=True)
            (models / "model.onnx").write_bytes(b"deployment-fixture")
            (models / "detector.json").write_text(json.dumps({"schema_version": 1, "model": "model.onnx"}))
            with (
                patch("deployment_support.PROJECT", root),
                patch.dict(
                    os.environ,
                    {
                        "HARBOUR_FL_DETECTOR_ENABLED": "true",
                        "HARBOUR_FL_DETECTOR_CONFIG": str(models / "detector.json"),
                    },
                ),
            ):
                native = detector_environment()
                container = detector_environment(container=True)
            self.assertEqual(native["HARBOUR_FL_DETECTOR_CONFIG"], str((models / "detector.json").resolve()))
            self.assertEqual(container["HARBOUR_FL_DETECTOR_CONFIG"], "/opt/models/vessel-detector/detector.json")
            self.assertEqual(container["HARBOUR_FL_DETECTOR_CACHE"], "/tmp/harbour-fl-detector-cache")

    def test_gpu_allocation_is_opt_in_and_only_applies_to_clients(self):
        """Verify gpu allocation is opt in and only applies to clients."""
        for count in (0, 1):
            with (
                tempfile.TemporaryDirectory() as directory,
                patch.dict("os.environ", {"HARBOUR_FL_K3S_GPUS_PER_CLIENT": str(count)}),
            ):
                path = Path(directory) / "k3s.yaml"
                documents = []
                for name in ("ws-server", "recovery-supervisor", "vessel-1", "vessel-coordinator"):
                    container = {
                        "env": [
                            {"name": "MODULES_ROOT", "value": ""},
                            {"name": "HARBOUR_FL_DATA_ROOT", "value": "/data"},
                        ]
                    }
                    documents.append(
                        {
                            "kind": "Deployment",
                            "metadata": {"name": name},
                            "spec": {"template": {"spec": {"containers": [container]}}},
                        }
                    )
                path.write_text(yaml.safe_dump_all(documents))
                configure_k3s(path)
                result = list(yaml.safe_load_all(path.read_text()))
                for item in result:
                    if item.get("kind") != "Deployment":
                        continue
                    container = item["spec"]["template"]["spec"]["containers"][0]
                    limit = container.get("resources", {}).get("limits", {}).get("nvidia.com/gpu", 0)
                    self.assertEqual(limit, count if item["metadata"]["name"] == "vessel-1" else 0)

    def test_credentials_survive_regeneration_and_are_shared_between_deployment_types(self):
        """Verify credentials survive regeneration and are shared between deployment types."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "deployment" / "secrets.env"
            second = root / "deployment-k3s" / "secrets.env"
            first.parent.mkdir()
            second.parent.mkdir()
            original = b"ZO_ROOT_USER_PASSWORD=example-first\nOTLP_AUTH_PASSWORD=example-first\n"
            first.write_bytes(original)
            retain_credentials(first)
            regenerated = b"ZO_ROOT_USER_PASSWORD=example-second\nOTLP_AUTH_PASSWORD=example-second\n"
            first.write_bytes(regenerated)
            second.write_bytes(regenerated)
            retain_credentials(first)
            retain_credentials(second)
            stable = root / "deployment-secrets" / "secrets.env"
            for path in (first, second, stable):
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_invalid_credentials_are_rejected_without_overwriting_generated_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = root / "deployment-k3s" / "secrets.env"
            stable = root / "deployment-secrets" / "secrets.env"
            generated.parent.mkdir()
            original = b"ZO_ROOT_USER_PASSWORD=example-first\nOTLP_AUTH_PASSWORD=example-first\n"
            generated.write_bytes(original)
            stable.parent.mkdir()
            stable.write_text("ZO_ROOT_USER_PASSWORD=example-only\n")
            with self.assertRaisesRegex(ValueError, "must be nonempty and match"):
                retain_credentials(generated)
            self.assertEqual(generated.read_bytes(), original)

    def run_registry_initializer(self, path):
        return subprocess.run(
            [sys.executable, "-c", REGISTRY_INITIALIZER, str(path), "vessel-1", "recovery-supervisor"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_new_registry_reserves_ids_with_valid_unspecified_addresses(self):
        """Verify new registry reserves ids with valid unspecified addresses."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.yaml"
            result = self.run_registry_initializer(path)
            self.assertEqual(result.returncode, 0, result.stderr)
            records = yaml.safe_load(path.read_text())
            for name in ("vessel-1", "recovery-supervisor"):
                self.assertEqual(records[name], {"state": "disconnected", "last_known_ip": "0.0.0.0"})
            self.assertFalse(path.with_name("registry.yaml.pre-ip-required").exists())

    def test_registry_migration_preserves_addresses_messages_and_original_backup(self):
        """Verify registry migration preserves addresses messages and original backup."""
        records = {
            "vessel-1": {"state": "disconnected", "last_known_ip": None, "pending_direct_messages": {"m": {"x": 1}}},
            "legacy-browser": {"state": "disconnected"},
            "ipv4": {"state": "disconnected", "last_known_ip": "10.42.0.12"},
            "ipv6": {"state": "disconnected", "last_known_ip": "2001:db8::12"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.yaml"
            path.write_text(yaml.safe_dump(records))
            original = path.read_bytes()
            result = self.run_registry_initializer(path)
            self.assertEqual(result.returncode, 0, result.stderr)
            expected = {name: dict(record) for name, record in records.items()}
            expected["vessel-1"]["last_known_ip"] = "0.0.0.0"
            expected["legacy-browser"]["last_known_ip"] = "0.0.0.0"
            expected["recovery-supervisor"] = {"state": "disconnected", "last_known_ip": "0.0.0.0"}
            self.assertEqual(yaml.safe_load(path.read_text()), expected)
            backup = path.with_name("registry.yaml.pre-ip-required")
            self.assertEqual(backup.read_bytes(), original)
            self.assertEqual(self.run_registry_initializer(path).returncode, 0)
            self.assertEqual(backup.read_bytes(), original)
            self.assertEqual(yaml.safe_load(path.read_text()), expected)

    def test_invalid_registry_data_is_not_silently_replaced(self):
        """Verify invalid registry data is not silently replaced."""
        invalid_documents = [
            {"vessel-1": {"state": "disconnected", "last_known_ip": value}}
            for value in ("unknown", "fe80::1%eth0", 1234, False)
        ] + [["not-a-mapping"], {"vessel-1": "not-a-record"}]
        for document in invalid_documents:
            with self.subTest(document=document), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "registry.yaml"
                path.write_text(yaml.safe_dump(document))
                original = path.read_bytes()
                result = self.run_registry_initializer(path)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("ValueError", result.stderr)
                self.assertEqual(path.read_bytes(), original)
                self.assertFalse(path.with_name("registry.yaml.pre-ip-required").exists())

    def test_recovery_is_added_only_to_the_generated_k3s_input(self):
        """Verify recovery is added only to the generated k3s input."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = (PROJECT / "scenario.yaml").read_text()
            (root / "scenario.yaml").write_text(original)
            (root / "recovery.json").write_bytes((PROJECT / "recovery.json").read_bytes())
            output = configure_scenario(root / "scenario.yaml")
            scenario = yaml.safe_load(output.read_text())
            bridge = next(agent for agent in scenario["agents"] if agent["name"] == "mlflow-bridge")
            self.assertEqual(bridge["env"]["PYO3_MLFLOW_ARTIFACT_BUCKETS"], "vessel-coordinator")
            self.assertEqual(scenario["deployment_type"], "k3s")
            self.assertEqual(len(scenario["recovery"]["workloads"]), 4)
            self.assertEqual(scenario["recovery"]["mode"], "observe-only")
            for workload in scenario["recovery"]["workloads"]:
                self.assertEqual(workload["agent_id"], workload["deployment"])
                self.assertEqual(workload["source"], "/modules/@edge-toolkit/harbour-fl-engine")
            self.assertEqual((root / "scenario.yaml").read_text(), original)

    def test_old_release_is_rejected_before_staging_or_deployment(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(RuntimeError, "Rebuild that release first"),
        ):
            prepare_release(Path(directory))

    def test_release_preflight_works_before_third_party_packages_are_installed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-S", str(PROJECT / "configure_deployment.py"), "core-release", directory],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Rebuild that release first", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_generated_core_recovery_is_not_replaced_by_a_research_controller(self):
        """Verify generated core recovery is not replaced by a research controller."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "k3s.yaml"
            documents = [
                {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "metadata": {"name": name},
                    "spec": {
                        "template": {
                            "spec": {
                                "containers": [
                                    {
                                        "env": [
                                            {"name": "MODULES_ROOT", "value": ""},
                                            {"name": "WS_AGENT_REGISTRY", "value": "/app/runtime/registry.yaml"},
                                        ]
                                    }
                                ]
                            }
                        }
                    },
                }
                for name in ("ws-server", "recovery-supervisor")
            ]
            documents.append({"kind": "Role", "metadata": {"name": "recovery-supervisor"}, "rules": []})
            path.write_text(yaml.safe_dump_all(documents))
            configure_k3s(path)
            result = list(yaml.safe_load_all(path.read_text()))
            supervisor = next(item for item in result if item.get("metadata", {}).get("name") == "recovery-supervisor")
            container = supervisor["spec"]["template"]["spec"]["containers"][0]
            self.assertEqual(container["image"], "et-recovery-supervisor:local")
            self.assertTrue(any(item.get("kind") == "Role" for item in result))
            hub = next(item for item in result if item.get("metadata", {}).get("name") == "ws-server")
            hub_container = hub["spec"]["template"]["spec"]["containers"][0]
            self.assertNotIn("args", hub_container)
            registry = [entry for entry in hub_container["env"] if entry["name"] == "WS_AGENT_REGISTRY"]
            self.assertEqual(registry, [{"name": "WS_AGENT_REGISTRY", "value": "/app/storage/registry.yaml"}])
            command = hub["spec"]["template"]["spec"]["initContainers"][0]["command"]
            self.assertEqual(command[2], REGISTRY_INITIALIZER)
            self.assertEqual(command[3], "/app/storage/registry.yaml")
            self.assertEqual(
                command[4:],
                ["vessel-1", "vessel-2", "vessel-3", "vessel-coordinator", "mlflow-bridge", "recovery-supervisor"],
            )
