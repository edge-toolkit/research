"""Exercise pressure profiles and cleanup using small allocations and local network fixtures."""

import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import Mock, patch

from et_experiments.benchmarking import memory_worker, network_worker
from et_experiments.benchmarking.saturation import (
    MEMORY_TYPE,
    NETWORK_TYPE,
    SaturationFaults,
    annotate_stages,
    profile_options,
    worker_source,
)
from et_experiments.contracts import WorkloadConfig
from et_experiments.tracking.mlflow import mlflow_benchmark_payload
from et_experiments.utils import BenchmarkError


class SaturationTests(unittest.TestCase):
    def test_remote_probe_allows_service_propagation_but_has_a_deadline(self):
        options = {"host": "private-service", "port": 19090, "token": "test"}
        peer = Mock()
        with patch.object(network_worker.socket, "socket") as factory:
            factory.return_value.__enter__.return_value = peer
            peer.recvfrom.side_effect = [ConnectionRefusedError(), (b"test" + bytes(12), ("127.0.0.1", 19090))]
            with patch.object(network_worker.time, "sleep"):
                network_worker.probe(options, remote=True)
            self.assertEqual(peer.sendto.call_count, 2)
            self.assertEqual(peer.sendto.call_args.args[1], ("private-service", 19090))
            peer.recvfrom.side_effect = TimeoutError()
            with patch.object(network_worker.time, "monotonic", side_effect=[0, 11]), self.assertRaises(TimeoutError):
                network_worker.probe(options, remote=True)

    def test_refused_udp_path_retains_measurements_without_claiming_congestion(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        report = network_worker.run_profile(
            {
                "host": "127.0.0.1",
                "port": port,
                "token": "test",
                "rate_kbit": 128,
                "hold_seconds": 0.02,
                "levels": [0.25, 1.5],
            }
        )
        self.assertEqual(report["status"], "not-observed")
        self.assertEqual(report["pressure_classification"], "probe-path-error")
        self.assertGreater(report["metrics"]["network_probe_transport_errors"], 0)
        self.assertEqual(report["metrics"]["pressure_observed"], 0)
        self.assertEqual(report["metrics"]["network_saturation_verified"], 0)
        self.assertEqual(len(report["stages"]), 2)

    def test_unreachable_receiver_is_rejected_before_applying_pressure(self):
        self.helper.netem = Mock(return_value=CompletedProcess([], 0, "qdisc noqueue 0: root"))
        self.helper.execute = Mock()
        self.backend.kubectl.side_effect = lambda *args, **_kwargs: CompletedProcess(
            [], 1 if "path-probe" in args else 0, "", "receiver unavailable"
        )
        with self.assertRaisesRegex(BenchmarkError, "unreachable before pressure"):
            self.helper.inject({"type": NETWORK_TYPE, "target": "client"}, "job")
        self.helper.netem.assert_called_once_with("client", "show", "dev", "eth0")
        self.helper.execute.assert_not_called()
        self.assertEqual(self.backend.kubectl.call_args.args[0], "delete")

    def test_shared_link_congestion_does_not_claim_a_verified_saturation_curve(self):
        samples = [
            {
                "offered_kbit": offered,
                "goodput_kbit": goodput,
                "ack_loss_ratio": loss,
                "rtt_p95_ms": rtt,
                "acknowledged_packets": 50,
            }
            for offered, goodput, loss, rtt in (
                (250, 170, 0.2, 1000),
                (501, 174, 0.5, 2512),
                (824, 86, 0.87, 3831),
                (1160, 154, 0.84, 2916),
                (1492, 277, 0.78, 1865),
            )
        ]
        with patch.object(network_worker, "stage", side_effect=samples):
            result = network_worker.run_profile(
                {
                    "host": "127.0.0.1",
                    "port": 19090,
                    "token": "test",
                    "rate_kbit": 1024,
                    "levels": [0.25, 0.5, 0.85, 1.25, 1.5],
                    "hold_seconds": 5,
                }
            )
        self.assertEqual(result["status"], "restored")
        self.assertEqual(result["pressure_classification"], "congestion-observed")
        self.assertEqual(result["metrics"]["pressure_observed"], 1)
        self.assertEqual(result["metrics"]["network_saturation_verified"], 0)
        self.assertFalse(result["saturation_evidence"]["probe_delivery_threshold_met"])

    def test_verified_saturation_remains_distinct_from_congestion_only(self):
        samples = [
            {
                "offered_kbit": 250,
                "goodput_kbit": 240,
                "ack_loss_ratio": 0,
                "rtt_p95_ms": 1,
                "acknowledged_packets": 50,
            },
            {
                "offered_kbit": 1200,
                "goodput_kbit": 950,
                "ack_loss_ratio": 0.1,
                "rtt_p95_ms": 30,
                "acknowledged_packets": 50,
            },
            {
                "offered_kbit": 1500,
                "goodput_kbit": 960,
                "ack_loss_ratio": 0.2,
                "rtt_p95_ms": 50,
                "acknowledged_packets": 50,
            },
        ]
        with patch.object(network_worker, "stage", side_effect=samples):
            result = network_worker.run_profile(
                {
                    "host": "127.0.0.1",
                    "port": 19090,
                    "token": "test",
                    "rate_kbit": 1024,
                    "levels": [0.25, 1.25, 1.5],
                    "hold_seconds": 5,
                }
            )
        self.assertEqual(result["pressure_classification"], "saturation-verified")
        self.assertEqual(result["metrics"]["network_saturation_verified"], 1)

    def setUp(self):
        self.backend = Mock(namespace="test")
        self.backend.kubectl.return_value = CompletedProcess([], 0, stdout="", stderr="")
        self.helper = SaturationFaults(self.backend, WorkloadConfig("ctx", "test", "main", ("client",), (), "python"))

    def test_bounds_reject_unsafe_or_unordered_profiles(self):
        for update in (
            {"levels": [0.9, 0.6]},
            {"levels": [0.6, 0.99]},
            {"levels": [0.6, float("nan")]},
            {"max_extra_memory_mib": 2048},
            {"hold_seconds": 200},
        ):
            with self.assertRaises(BenchmarkError):
                profile_options({"type": MEMORY_TYPE, **update})
        with self.assertRaises(BenchmarkError):
            self.helper.inject({"type": NETWORK_TYPE, "target": "outside"}, "job")
        self.backend.kubectl.assert_not_called()

    def test_memory_reader_requires_a_finite_container_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "memory.current").write_text("1048576")
            (root / "memory.max").write_text("max")
            with self.assertRaisesRegex(ValueError, "finite"):
                memory_worker.memory_snapshot(root)
            (root / "memory.max").write_text("268435456")
            self.assertEqual(memory_worker.memory_snapshot(root), (1048576, 268435456))

    def test_memory_uses_observed_usage_and_respects_the_extra_allocation_cap(self):
        snapshots = iter((value, 100000) for value in [20000, 40000, 60000, 80000, 90000, 25000])
        clock = [0.0]

        def sleep(seconds):
            clock[0] += seconds

        options = {
            "levels": [0.6, 0.9],
            "hold_seconds": 0.01,
            "max_extra_memory_mib": 0.001,
            "reserve_mib": 0.00001,
            "release_seconds": 0,
        }
        report = memory_worker.run_profile(options, lambda: next(snapshots), lambda: clock[0], sleep)
        self.assertEqual(report["status"], "restored")
        self.assertAlmostEqual(report["metrics"]["memory_utilization_peak"], 0.9)
        self.assertLessEqual(report["metrics"]["memory_extra_peak_mib"], 0.001)
        self.assertEqual(report["after_release_memory_bytes"], 25000)

    def test_memory_safety_stop_never_reports_saturation_success(self):
        report = memory_worker.run_profile(
            {
                "levels": [0.6, 0.9],
                "hold_seconds": 1,
                "max_extra_memory_mib": 0.001,
                "reserve_mib": 0.00001,
                "release_seconds": 0,
            },
            lambda: (96000, 100000),
            sleep=lambda _seconds: None,
        )
        self.assertTrue(report["safety_stop"])
        self.assertEqual(report["status"], "not-observed")
        self.assertEqual(report["metrics"]["pressure_observed"], 0)

    def test_reference_memory_resources_are_restored_exactly(self):
        resources = {"limits": {"cpu": "2", "memory": "2Gi"}, "requests": {"cpu": "100m", "memory": "128Mi"}}
        self.backend.kubectl.return_value.stdout = json.dumps(
            {"spec": {"template": {"spec": {"containers": [{"name": "client", "resources": resources}]}}}}
        )
        self.helper.prepare({"type": MEMORY_TYPE, "target": "client", "memory_limit_mib": 768})
        patch = json.loads(self.backend.kubectl.call_args.args[-1])
        self.assertEqual(patch[0]["value"]["limits"]["memory"], "768Mi")
        self.helper.restore()
        self.assertEqual(json.loads(self.backend.kubectl.call_args.args[-1])[0]["value"], resources)
        self.assertFalse(self.helper.original_resources)

    def test_network_cleanup_runs_when_the_traffic_worker_fails(self):
        self.helper.netem = Mock(
            side_effect=[
                CompletedProcess([], 0, "qdisc noqueue 0: root"),
                CompletedProcess([], 0, ""),
                CompletedProcess([], 0, ""),
            ]
        )
        self.helper.execute = Mock(side_effect=BenchmarkError("worker failed"))
        with self.assertRaisesRegex(BenchmarkError, "worker failed"):
            self.helper.inject({"type": NETWORK_TYPE, "target": "client"}, "job")
        self.assertEqual(self.helper.netem.call_args.args, ("client", "del", "dev", "eth0", "root"))
        self.assertEqual(self.backend.kubectl.call_args.args[0], "delete")

    def test_existing_traffic_policy_is_not_overwritten(self):
        self.helper.netem = Mock(return_value=CompletedProcess([], 0, "qdisc fq_codel 0: root"))
        with self.assertRaisesRegex(BenchmarkError, "pre-existing"):
            self.helper.inject({"type": NETWORK_TYPE, "target": "client"}, "job")
        self.helper.netem.assert_called_once()
        self.assertEqual(self.backend.kubectl.call_args.args[0], "delete")

    def test_receiver_cleanup_is_attempted_even_if_qdisc_cleanup_raises(self):
        self.helper.netem = Mock(
            side_effect=[
                CompletedProcess([], 0, "qdisc noqueue 0: root"),
                CompletedProcess([], 0, ""),
                BenchmarkError("cleanup failed"),
            ]
        )
        self.helper.execute = Mock(return_value={"status": "restored"})
        with self.assertRaisesRegex(BenchmarkError, "cleanup failed"):
            self.helper.inject({"type": NETWORK_TYPE, "target": "client"}, "job")
        self.assertEqual(self.backend.kubectl.call_args.args[0], "delete")

    def test_complete_packet_loss_is_not_misreported_as_saturation(self):
        samples = [
            {
                "offered_kbit": 3000,
                "goodput_kbit": 0,
                "ack_loss_ratio": 1,
                "rtt_p95_ms": None,
                "acknowledged_packets": 0,
            }
            for _ in range(2)
        ]
        with patch.object(network_worker, "stage", side_effect=samples):
            result = network_worker.run_profile(
                {
                    "host": "127.0.0.1",
                    "port": 19090,
                    "token": "test",
                    "levels": [0.25, 1.5],
                    "rate_kbit": 1024,
                    "hold_seconds": 1,
                }
            )
        self.assertEqual(result["status"], "not-observed")

    def test_stage_metrics_do_not_turn_missing_application_timestamps_into_zero_throughput(self):
        stage = {"started_at": "2026-01-01T00:00:00+00:00", "finished_at": "2026-01-01T00:00:10+00:00"}
        report = {"fault": {"stages": [stage]}, "workload": {"units": [{"duration_seconds": 2}]}}
        annotate_stages(report)
        self.assertIsNone(stage["application"]["completed_units_per_second"])
        report["workload"]["units"][0]["completed_at"] = "2026-01-01T00:00:05+00:00"
        annotate_stages(report)
        self.assertEqual(stage["application"]["completed_units_per_second"], 0.1)
        self.assertEqual(stage["application"]["unit_duration_p95_seconds"], 2)

    def test_receiver_is_private_unprivileged_and_time_bounded(self):
        manifest = self.helper.receiver("receiver", {"token": "test", "port": 19090, "lifetime_seconds": 150})
        pod, service = manifest["items"]
        self.assertFalse(pod["spec"].get("hostNetwork", False))
        self.assertEqual(pod["spec"]["activeDeadlineSeconds"], 180)
        self.assertTrue(pod["spec"]["securityContext"]["runAsNonRoot"])
        self.assertNotEqual(service["spec"].get("type"), "NodePort")
        self.assertIn("readinessProbe", pod["spec"]["containers"][0])

    def test_actual_receiver_program_supports_readiness_and_profile_headers(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        options = {
            "host": "127.0.0.1",
            "port": port,
            "token": "test-token",
            "lifetime_seconds": 8,
            "rate_kbit": 128,
            "hold_seconds": 0.1,
            "levels": [0.25, 0.5],
        }
        script = worker_source("network_worker.py")
        child = subprocess.Popen([sys.executable, "-c", script, "serve", json.dumps(options)])
        try:
            deadline = time.monotonic() + 4
            while True:
                try:
                    network_worker.probe(options)
                    break
                except (TimeoutError, ConnectionRefusedError):
                    if time.monotonic() >= deadline:
                        raise
            network_worker.probe(options, remote=True)
            measured = subprocess.run(
                [sys.executable, "-c", script, "run", json.dumps(options)],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            report = json.loads(measured.stdout)
            self.assertTrue(all(stage["acknowledged_packets"] > 0 for stage in report["stages"]))
            self.assertEqual(report["status"], "not-observed")
        finally:
            child.terminate()
            child.wait(timeout=3)

    def test_pressure_summaries_are_logged_without_creating_a_metric_per_sample(self):
        payload = mlflow_benchmark_payload(
            {"fault": {"metrics": {"memory_utilization_peak": 0.9, "pressure_observed": 1}}}, "pressure"
        )
        self.assertEqual(payload["metrics"]["benchmark.pressure.memory_utilization_peak"], 0.9)
        self.assertEqual(payload["metrics"]["benchmark.pressure.pressure_observed"], 1)

    def test_udp_measurements_use_real_echoes_not_requested_rate(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
            server.bind(("127.0.0.1", 0))
            server.settimeout(0.1)
            stop = threading.Event()

            def echo():
                while not stop.is_set():
                    try:
                        payload, address = server.recvfrom(1500)
                        server.sendto(payload[:16], address)
                    except TimeoutError:
                        continue

            thread = threading.Thread(target=echo)
            thread.start()
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
                    client.connect(server.getsockname())
                    client.setblocking(False)
                    result = network_worker.stage(client, b"token123", 128, 0.1, drain_seconds=0.02)
                self.assertGreater(result["acknowledged_packets"], 0)
                self.assertGreater(result["goodput_kbit"], 0)
                self.assertIsNotNone(result["rtt_p95_ms"])
            finally:
                stop.set()
                thread.join(timeout=2)
