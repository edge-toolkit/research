import datetime as dt
import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from et_experiments.benchmarking import engine as engine_module
from et_experiments.benchmarking.engine import BenchmarkEngine
from et_experiments.benchmarking.faults import memory_pressure_manifest, network_fault_arguments
from et_experiments.benchmarking.observations import recovery_outcome
from et_experiments.benchmarking.sampling import resource_summary
from et_experiments.tracking.mlflow import mlflow_benchmark_payload
from et_experiments.utils import BenchmarkError, iso, parse_cpu, parse_memory, percentile, utc_now
from et_experiments.web.controller import BenchmarkController, ConfigurationError
from harbour_profile import application_metrics, observe_training, tracking_recovery, training_finished
from integration import CONTEXT, HarbourFLAdapter, application_fault, parse_logs

SCENARIOS_PATH = Path(__file__).with_name("scenarios.json")


def patch_dependency(engine, name, replacement):
    """Patch the engine component that owns a named backend, tracking or monitoring dependency."""
    backend_names = {"kubectl", "primary_logs", "collect_logs", "warning_events"}
    if name in backend_names:
        return patch.object(engine.backend, name, replacement)
    if name in {"mlflow_state", "annotate_mlflow"}:
        return patch.object(engine.tracking, name, replacement)
    if name == "inject_fault":
        return patch.object(engine.faults, name, replacement)
    if name == "progress":
        return patch.object(engine_module, name, replacement)
    names = {
        "Sampler": "sampler_factory",
        "CoreRecoveryRuntime": "runtime_factory",
        "CoreRecoveryMonitor": "monitor_factory",
    }
    return patch.object(engine, names.get(name, name), replacement)


def add_workload_events(logs, job_id):
    """Add matching workload-identity events for all four application roles to synthetic logs."""
    from harbour_fl.workload import signature

    for name in ("vessel-1", "vessel-2", "vessel-3", "vessel-coordinator"):
        event = {
            "schema_version": 1,
            "id": job_id + name,
            "instance": name,
            "kind": "workload-profile",
            "job_id": job_id,
            "workload": name,
            "at": utc_now().timestamp(),
            "settings": signature(),
        }
        logs[name] = logs.get(name, "") + "\nET_EVENT " + json.dumps(event)


class BenchmarkParserTests(unittest.TestCase):
    def test_service_probe_check_uses_the_runner_interpreter_and_named_container(self):
        """Verify service probe check uses the runner interpreter and named container."""
        backend = Mock()
        backend.kubectl.return_value = Mock(returncode=0)
        with patch("integration.HubTaskProbeTransport"), patch("integration.ServiceProbeMonitor") as monitor:
            result = HarbourFLAdapter().service_probe_monitor(backend, "job", {"service_probes": {"enabled": True}})
        self.assertIs(result, monitor.return_value)
        self.assertEqual(backend.kubectl.call_count, 4)
        for call in backend.kubectl.call_args_list:
            args = call.args
            self.assertEqual(args[args.index("--") + 1], "/opt/cpython/bin/python3")
            self.assertEqual(args[args.index("-c") + 1], args[1].split("/")[1])

    def test_service_probe_check_preserves_the_underlying_failure(self):
        backend = Mock()
        backend.kubectl.return_value = Mock(returncode=1, stderr="interpreter not found", stdout="")
        with self.assertRaisesRegex(BenchmarkError, "vessel-1 service-probe check failed: interpreter not found"):
            HarbourFLAdapter().service_probe_monitor(backend, "job", {"service_probes": {"enabled": True}})

    def test_reference_conditions_capture_workload_and_manifest_versions(self):
        adapter = HarbourFLAdapter()
        reference = adapter.reference_conditions({"rounds": 12, "checkpoint_interval": 3, "minimum_clients": 2})
        self.assertEqual(len(reference["clients"]), 3)
        self.assertEqual(reference["checkpoint_interval"], 3)
        self.assertEqual(reference["minimum_clients"], 2)
        self.assertEqual(len(reference["application_source_sha256"]), 64)
        self.assertEqual(len(reference["dataset_manifests"]), 8)

    def test_new_profiles_preserve_three_clients_and_have_bounded_workloads(self):
        """Verify new profiles preserve three clients and have bounded workloads."""
        from et_experiments.benchmarking.design import FaultTrigger, validate_design

        scenarios = json.loads(SCENARIOS_PATH.read_text())
        adapter = HarbourFLAdapter()
        for name in (
            "reference-training-outage",
            "reference-client-stage-restart",
            "checkpoint-recovery-recent",
            "checkpoint-recovery-stale",
            "checkpoint-recovery-precommit",
            "soak-baseline",
            "soak-client-outage",
            "combined-client-network-outage",
        ):
            with self.subTest(name=name):
                scenario = adapter.prepare_scenario(scenarios[name])
                validate_design(scenario, adapter.config.workers)
                FaultTrigger(scenario["fault"])
                self.assertGreaterEqual(scenario["rounds"], 12)
                self.assertLessEqual(scenario["timeout_seconds"], 7200)

    def test_invalid_training_and_checkpoint_controls_fail_before_setup(self):
        """Verify invalid training and checkpoint controls fail before setup."""
        for setting in (
            {"checkpoint_interval": 0},
            {"minimum_clients": 4},
            {"batch_size": 0},
            {"checkpoint_write_delay_seconds": float("nan")},
        ):
            with self.subTest(setting=setting), self.assertRaises(BenchmarkError):
                HarbourFLAdapter().prepare_scenario({"rounds": 3, **setting})

    def test_saturation_presets_require_post_release_progress_and_keep_oom_separate(self):
        """Verify saturation presets require post release progress and keep oom separate."""
        scenarios = json.loads(SCENARIOS_PATH.read_text())
        adapter = HarbourFLAdapter()
        for name in ("client-memory-saturation", "client-network-saturation"):
            scenario = adapter.prepare_scenario(scenarios[name])
            self.assertTrue(scenario["require_post_fault_progress"])
            self.assertEqual(scenario["fault"]["target"], "vessel-2")
            self.assertEqual(scenario["fault"]["after_work_unit"], 0)
            self.assertGreaterEqual(scenario["rounds"], 10)
        self.assertEqual(scenarios["client-oom"]["fault"]["type"], "client_oom")

    def test_gpu_catalog_requires_accelerators_and_keeps_cpu_baseline(self):
        """Verify gpu catalog requires accelerators and keeps cpu baseline."""
        scenarios = json.loads(SCENARIOS_PATH.read_text())
        adapter = HarbourFLAdapter()
        self.assertEqual(
            adapter.prepare_scenario(scenarios["gpu-runtime-failure"])["client_devices"], {"vessel-2": "cuda"}
        )
        self.assertEqual(scenarios["baseline"].get("device", "cpu"), "cpu")
        for device in ("cpu", "auto", "mps"):
            with self.assertRaises(BenchmarkError):
                adapter.prepare_scenario({"device": device, "fault": {"type": "gpu_failure"}})

    def test_adapter_translates_domain_faults_without_mutating_catalog_scenarios(self):
        original = {"rounds": 3, "fault": {"type": "client_oom", "after_round": 0}}
        adapter = HarbourFLAdapter()
        prepared = adapter.prepare_scenario(original)
        self.assertEqual(prepared["fault"], {"type": "worker_oom", "after_work_unit": 0})
        self.assertEqual(original["fault"], {"type": "client_oom", "after_round": 0})
        self.assertEqual(adapter.expected_units(prepared), 3)

    def test_adapter_maps_rounds_to_work_units_and_owns_domain_report_fields(self):
        adapter = HarbourFLAdapter()
        logs = {
            "vessel-coordinator": "coordinator completed round 0: "
            "{'global_test_accuracy': 0.5, 'participating_clients': 3}"
        }
        observed = adapter.observe(logs, None, 1, dt.datetime.now(dt.timezone.utc))
        self.assertEqual(observed["completed_units"], 1)
        report = {
            "workload": observed,
            "wall_time_seconds": 2,
            "fault": {"type": "none"},
            "events": [],
            "resilience": {"completion_ratio": 1, "completed_units_per_minute": 30},
        }
        slis = adapter.enrich_report(report)
        self.assertEqual(slis["model_accuracy"], 0.5)
        self.assertEqual(report["resilience"]["mean_client_participation_ratio"], 1)
        self.assertEqual(report["application_metrics"]["completed_rounds_per_minute"], 30)
        self.assertIs(report["training"], observed["details"])

    def test_tracking_readiness_is_not_ingestion_recovery(self):
        """Verify tracking readiness is not ingestion recovery."""
        fault = {"started_at": "1970-01-01T00:01:40Z"}
        state = {"status": "FINISHED", "history": {"loss": [{"step": 0, "timestamp": 50000}]}}
        self.assertEqual(tracking_recovery(state, 3, "loss", fault)["status"], "pending")
        state["history"]["loss"] = [{"step": i, "timestamp": 101000} for i in range(3)]
        result = tracking_recovery(state, 3, "loss", fault, verified_at=110)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["recovery_seconds"], 10)
        self.assertFalse(result["forced_bridge_restart"])
        self.assertEqual(result["recovery_actor"], "core-mlflow-bridge")
        fault["interventions"] = [{"type": "bridge-restart"}]
        self.assertEqual(tracking_recovery(state, 3, "loss", fault)["recovery_actor"], "benchmark-assisted-bridge")

    def test_catalog_exposes_explicit_recovery_modes(self):
        """Verify catalog exposes explicit recovery modes."""
        controller = BenchmarkController(SCENARIOS_PATH.parents[1] / "benchmark-catalog.json")
        for name, expected in (
            ("recovery-slo", "observe-only"),
            ("recovery-slo-assisted", "controller-assisted"),
            ("checkpoint-write-failure", "disabled"),
            ("ws-server-restart", "disabled"),
        ):
            _, values = controller.validate_request({"scenario": name})
            self.assertEqual(values["recovery_mode"], expected)
        with self.assertRaises(ConfigurationError):
            controller.validate_request(
                {
                    "scenario": "checkpoint-write-failure",
                    "parameters": {
                        "recovery_mode": "controller-assisted",
                    },
                }
            )

    @staticmethod
    def tracking_state(status="RUNNING"):
        return {
            "status": status,
            "run_id": "test-run",
            "history": {"global_test_loss": [{"step": 0, "value": 1.2, "timestamp": 1792108800000}]},
        }

    def test_checkpoint_failure_overrides_running_tracking_metrics(self):
        logged = parse_logs(
            {
                "vessel-coordinator": "coordinator completed round 0: {'global_test_loss': 1.2}\n"
                "ERROR harbour FL coordinator failed\nOSError: injected checkpoint write failure for round 1"
            },
            3,
        )
        state = self.tracking_state()
        result = observe_training(logged, state, 3, dt.datetime.now(dt.timezone.utc))
        self.assertTrue(result["coordinator_failed"])
        self.assertEqual(result["completed_rounds"], 1)
        self.assertTrue(training_finished(result, state, recovery_enabled=False))
        self.assertFalse(training_finished(result, state, recovery_enabled=True))
        self.assertFalse(training_finished(result, {"status": "FINISHED"}, recovery_enabled=True))

    def test_resumed_progress_clears_stale_tracking_failure(self):
        logged = parse_logs(
            {
                "vessel-coordinator": "ERROR harbour FL coordinator failed\n"
                "coordinator completed round 1: {'global_test_loss': 1.0}"
            },
            3,
        )
        state = self.tracking_state("FAILED")
        result = observe_training(logged, state, 3, dt.datetime.now(dt.timezone.utc))
        self.assertFalse(result["coordinator_failed"])
        self.assertEqual(result["completed_rounds"], 2)
        self.assertFalse(training_finished(result, state, recovery_enabled=True))

    def test_terminal_tracking_without_metrics_is_not_ignored(self):
        """Verify terminal tracking without metrics is not ignored."""
        logged = parse_logs({}, 3)
        for status in ("FAILED", "KILLED"):
            state = {"status": status, "history": {}}
            result = observe_training(logged, state, 3, dt.datetime.now(dt.timezone.utc))
            self.assertTrue(result["coordinator_failed"])
            self.assertTrue(training_finished(result, state, recovery_enabled=False))

    def test_checkpoint_failure_finishes_and_reports_without_waiting_for_mlflow(self):
        """Verify checkpoint failure finishes and reports without waiting for mlflow."""
        engine = BenchmarkEngine(HarbourFLAdapter())
        scenario = json.loads(SCENARIOS_PATH.read_text())["checkpoint-write-failure"]
        for reason, expected_status in (
            ("injected checkpoint write failure for round 1", "passed"),
            ("unrelated training error", "failed"),
        ):
            logs = {
                "vessel-coordinator": "coordinator completed round 0: {'global_test_loss': 1.2}\n"
                f"ERROR harbour FL coordinator failed\nOSError: {reason}"
            }
            sampler = Mock(samples=[], metrics_available=False)
            replacements = {
                "kubectl": Mock(return_value=Mock(stdout=CONTEXT)),
                "configure_run": Mock(
                    side_effect=lambda job_id, _scenario, _started, logs=logs: add_workload_events(logs, job_id)
                ),
                "Sampler": Mock(return_value=sampler),
                "CoreRecoveryRuntime": Mock(return_value=Mock(events={})),
                "primary_logs": Mock(return_value=logs["vessel-coordinator"]),
                "collect_logs": Mock(return_value=logs),
                "mlflow_state": Mock(return_value=self.tracking_state()),
                "inject_fault": Mock(return_value={"type": "checkpoint_write_failure", "status": "configured"}),
                "warning_events": Mock(return_value=[]),
                "annotate_mlflow": Mock(return_value="test-run"),
                "restore": Mock(),
                "progress": Mock(),
            }
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as output, ExitStack() as stack:
                for name, replacement in replacements.items():
                    stack.enter_context(patch_dependency(engine, name, replacement))
                stack.enter_context(
                    patch.object(engine_module.time, "sleep", side_effect=AssertionError("unexpected wait"))
                )
                stack.enter_context(patch.dict(os.environ, {"BENCHMARK_RESULT_FILE": ""}))
                report = engine.execute("checkpoint-write-failure", scenario, Path(output), 2, False)
                self.assertEqual(report["status"], expected_status)
                self.assertTrue(report["training"]["coordinator_failed"])
                self.assertEqual(report["expected_failure_observed"], expected_status == "passed")
                self.assertEqual(report["training"]["completed_rounds"], 1)
                self.assertEqual(report["recovery"]["incidents"], [])
                replacements["restore"].assert_called_once()

    def test_resource_units(self):
        self.assertEqual(parse_cpu("250m"), 250)
        self.assertEqual(parse_cpu("2"), 2000)
        self.assertEqual(parse_memory("1Gi"), 1024)
        self.assertEqual(parse_memory("512Mi"), 512)

    def test_recovery_slo_waits_after_training_and_mlflow_finish(self):
        """Verify recovery slo waits after training and mlflow finish."""
        engine = BenchmarkEngine(HarbourFLAdapter())
        scenario = json.loads(SCENARIOS_PATH.read_text())["recovery-slo"]
        scenario["rounds"] = 3
        target = "vessel-coordinator"
        logs = {target: "\n".join(f"coordinator completed round {i}: {{'global_test_loss': 1.2}}" for i in range(3))}
        onset = utc_now().timestamp()
        incident = {"workload": target, "status": "recovering", "observed_failure_at": onset, "recovered_at": None}
        recovered = {**incident, "status": "recovered", "recovered_at": onset + 10}
        events = {
            kind: {"kind": kind, "workload": target, "at": onset + 9, "seconds": 0.1, "bytes": 100}
            for kind in ("checkpoint-restored", "resume-completed")
        }
        runtime = Mock(events=events)
        monitor = Mock()
        monitor.snapshot.side_effect = [
            {"incidents": [], "events": []},
            {"incidents": [incident], "events": []},
            {"incidents": [recovered], "events": list(events.values())},
        ]
        monitor.finish.return_value = {
            "incidents": [recovered],
            "metrics": {"mttr_seconds": 10, "recovery_success_ratio": 1},
        }
        replacements = {
            "kubectl": Mock(return_value=Mock(stdout=CONTEXT)),
            "configure_run": Mock(
                side_effect=lambda job_id, _scenario, _started, logs=logs: add_workload_events(logs, job_id)
            ),
            "Sampler": Mock(return_value=Mock(samples=[], metrics_available=False)),
            "CoreRecoveryRuntime": Mock(return_value=runtime),
            "CoreRecoveryMonitor": Mock(return_value=monitor),
            "primary_logs": Mock(return_value=logs[target]),
            "collect_logs": Mock(return_value=logs),
            "mlflow_state": Mock(return_value={"status": "FINISHED", "history": {}}),
            "inject_fault": Mock(
                return_value={
                    "type": "rollout_restart",
                    "target": target,
                    "status": "injected",
                    "started_at": iso(dt.datetime.fromtimestamp(onset, dt.timezone.utc)),
                }
            ),
            "warning_events": Mock(return_value=[]),
            "annotate_mlflow": Mock(return_value="test-run"),
            "restore": Mock(),
            "progress": Mock(),
        }
        with tempfile.TemporaryDirectory() as output, ExitStack() as stack:
            for name, replacement in replacements.items():
                stack.enter_context(patch_dependency(engine, name, replacement))
            sleep = stack.enter_context(patch.object(engine_module.time, "sleep"))
            stack.enter_context(patch.dict(os.environ, {"BENCHMARK_RESULT_FILE": ""}))
            report = engine.execute("recovery-slo", scenario, Path(output), 2, False)
            sleep.assert_called_once_with(2)
            self.assertEqual(monitor.snapshot.call_count, 3)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["slos"]["mttr_seconds"]["status"], "passed")
            self.assertEqual(report["slos"]["checkpoint_load_mean_seconds"]["status"], "passed")
            replacements["restore"].assert_called_once()

    def test_recovery_requires_post_fault_checkpoint_evidence(self):
        """Verify recovery requires post fault checkpoint evidence."""
        fault = {"target": "worker", "started_at": "1970-01-01T00:01:40Z"}
        scenario = {"require_checkpoint_restore": True}
        report = {"incidents": [{"workload": "worker", "status": "recovered"}], "events": []}
        self.assertEqual(recovery_outcome(report, scenario, fault), "pending")
        report["events"] = [
            {"workload": "worker", "kind": kind, "at": 99} for kind in ("checkpoint-restored", "resume-completed")
        ]
        self.assertEqual(recovery_outcome(report, scenario, fault), "pending")
        for event in report["events"]:
            event["at"] = 110
        self.assertEqual(recovery_outcome(report, scenario, fault), "recovered")
        report["incidents"][0]["status"] = "unrecovered"
        self.assertEqual(recovery_outcome(report, scenario, fault), "unrecovered")

    def test_log_metrics_and_round_duration(self):
        logs = {
            "vessel-coordinator": """2026-09-29T00:00:00Z INFO coordinator auto-start enabled
2026-09-29T00:00:01Z INFO coordinator checkpoint write round 0 completed in 0.125000 seconds
2026-09-29T00:00:10Z INFO coordinator completed round 0: {'global_test_accuracy': 0.5, 'global_test_loss': 1.2}
2026-09-29T00:00:25Z INFO coordinator completed round 1: {'global_test_accuracy': 0.7, 'global_test_macro_f1': 0.6}""",
            "vessel-1": "2026-09-29T00:00:08Z INFO vessel_1 completed round 0: {'train_accuracy': 0.8}",
        }
        result = parse_logs(logs, 2)
        self.assertEqual(result["completed_rounds"], 2)
        self.assertEqual(result["rounds"][0]["duration_seconds"], 10)
        self.assertEqual(result["rounds"][1]["duration_seconds"], 15)
        self.assertEqual(result["final_model_metrics"]["global_test_accuracy"], 0.7)
        self.assertEqual(result["client_rounds"]["vessel_1"][0]["round"], 0)
        self.assertEqual(result["checkpoint_write_seconds"], [0.125])

    def test_percentile(self):
        self.assertEqual(percentile([1, 2, 3, 4], 0.95), 4)
        self.assertIsNone(percentile([], 0.5))

    def test_resource_summary_tracks_replacement_and_availability(self):
        samples = [
            {"pods": {"vessel-2": {"name": "old", "ready": True, "restarts": 0, "network_tx_bytes": 100}}},
            {"pods": {"vessel-2": {"name": "new", "ready": False, "restarts": 0, "network_tx_bytes": 20}}},
            {"pods": {"vessel-2": {"name": "new", "ready": True, "restarts": 0, "network_tx_bytes": 70}}},
        ]
        result = resource_summary(samples)["vessel-2"]
        self.assertEqual(result["pod_replacements"], 1)
        self.assertEqual(result["not_ready_samples"], 1)
        self.assertAlmostEqual(result["observed_availability_ratio"], 2 / 3)
        self.assertEqual(result["network_tx_bytes"], 50)

    def test_network_fault_arguments(self):
        self.assertEqual(
            network_fault_arguments({"type": "network_latency", "latency_ms": 250, "jitter_ms": 50}),
            ("delay", "250ms", "50ms", "distribution", "normal"),
        )
        self.assertEqual(
            network_fault_arguments({"type": "network_bandwidth", "rate_kbit": 256}),
            ("rate", "256kbit"),
        )
        self.assertEqual(
            network_fault_arguments({"type": "network_loss", "loss_percent": 10}),
            ("loss", "10%"),
        )
        self.assertEqual(
            network_fault_arguments({"type": "network_partition"}),
            ("loss", "100%"),
        )

    def test_network_fault_values_are_bounded(self):
        self.assertEqual(
            network_fault_arguments({"type": "network_latency", "latency_ms": 99999, "jitter_ms": 99999}),
            ("delay", "5000ms", "5000ms", "distribution", "normal"),
        )

    def test_data_quality_faults_are_client_scoped_and_bounded(self):
        """Verify data quality faults are client scoped and bounded."""
        noisy = application_fault(
            {"type": "audio_noise", "target": "vessel-2", "snr_db": 10},
        )
        self.assertEqual(noisy["status"], "configured")
        self.assertEqual(noisy["snr_db"], 10)
        for invalid in ({"noise_std": 99}, {"snr_db": 5}, {"snr_db": float("nan")}):
            with self.subTest(fault=invalid), self.assertRaises(BenchmarkError):
                application_fault({"type": "audio_noise", "target": "vessel-2", **invalid})
        missing = application_fault(
            {"type": "missing_modality", "target": "vessel-2", "modality": "image"},
        )
        self.assertEqual(missing["modality"], "image")
        with self.assertRaises(BenchmarkError):
            application_fault(
                {"type": "missing_modality", "target": "mlflow", "modality": "audio"},
            )

    def test_memory_pressure_manifest_is_bounded_and_touches_memory(self):
        manifest = memory_pressure_manifest("bench-test-12345678", 9999, 30, "runner:test", "test")
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(container["args"][1], "2048")
        self.assertEqual(container["resources"]["limits"]["memory"], "2176Mi")
        self.assertIn("range(0,size,4096)", container["args"][0])
        self.assertTrue(container["securityContext"]["runAsNonRoot"])

    def test_mlflow_payload_keeps_metrics_compact_and_fault_details_as_tags(self):
        report = {
            "job_id": "bench-client-oom-test",
            "wall_time_seconds": 12.5,
            "fault": {
                "type": "client_oom",
                "target": "vessel-2",
                "status": "restored",
                "memory_limit_mib": 96,
                "failure_detection_seconds": 2.5,
                "injection_duration_seconds": 3.2,
                "termination": {"reason": "OOMKilled"},
            },
            "resilience": {
                "recovery_seconds": 3.1,
                "round_completion_ratio": 1.0,
                "mean_client_participation_ratio": 1.0,
                "completed_rounds_per_minute": 2.9,
            },
            "training": {
                "round_duration_seconds": {"mean": 3.5, "p50": 3.4, "p95": 3.8, "max": 3.8},
                "coordinator_failed": True,
                "checkpoint_write_seconds": [0.1, 0.2],
                "client_rounds": {
                    "vessel_1": [
                        {"metrics": {"dataset_read_mean_ms": 2, "dataset_read_p95_ms": 3, "samples_per_second": 40}}
                    ],
                    "vessel_2": [
                        {"metrics": {"dataset_read_mean_ms": 6, "dataset_read_p95_ms": 8, "samples_per_second": 20}}
                    ],
                },
            },
            "total_pod_restarts": 1,
            "total_pod_replacements": 1,
            "warning_events": [{"reason": "BackOff"}],
            "resources": {
                "vessel-1": {"cpu_millicores_peak": 200, "memory_mib_peak": 180, "network_rx_bytes": 100},
                "vessel-2": {"cpu_millicores_peak": 300, "memory_mib_peak": 240, "network_rx_bytes": 150},
            },
        }
        report["application_metrics"] = application_metrics(report)
        payload = mlflow_benchmark_payload(report, "client-oom")
        self.assertEqual(payload["metrics"]["benchmark.failure_detection_seconds"], 2.5)
        self.assertEqual(payload["metrics"]["benchmark.cpu_peak_millicores"], 300)
        self.assertEqual(payload["metrics"]["benchmark.network_rx_bytes"], 250)
        self.assertEqual(payload["metrics"]["benchmark.dataset_read_mean_ms"], 4)
        self.assertEqual(payload["metrics"]["benchmark.dataset_read_p95_ms"], 8)
        self.assertEqual(payload["metrics"]["benchmark.samples_per_second"], 30)
        self.assertEqual(payload["metrics"]["benchmark.checkpoint_write_p95_seconds"], 0.2)
        self.assertEqual(payload["metrics"]["benchmark.checkpoint_write_failures"], 0)
        self.assertNotIn("benchmark.memory_p95_mib", payload["metrics"])
        self.assertEqual(payload["tags"]["benchmark.termination_reason"], "OOMKilled")
        self.assertEqual(payload["tags"]["benchmark.fault_memory_limit_mib"], "96")


if __name__ == "__main__":
    unittest.main()
