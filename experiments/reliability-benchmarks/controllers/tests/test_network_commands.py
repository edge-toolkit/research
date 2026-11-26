"""Network shaping must support fixed delay as well as latency with jitter."""

import unittest

from et_experiments.benchmarking.faults import network_fault_arguments


class NetworkCommandTests(unittest.TestCase):
    def test_fixed_delay_omits_distribution(self):
        self.assertEqual(
            network_fault_arguments({"type": "network_latency", "latency_ms": 500, "jitter_ms": 0}),
            ("delay", "500ms"),
        )

    def test_combined_fixed_delay_preserves_loss_and_bandwidth(self):
        self.assertEqual(
            network_fault_arguments(
                {"type": "degraded_network", "latency_ms": 500, "jitter_ms": 0, "loss_percent": 30, "rate_kbit": 10}
            ),
            ("delay", "500ms", "loss", "30%", "rate", "10kbit"),
        )

    def test_positive_jitter_keeps_distribution(self):
        for kind in ("network_latency", "degraded_network"):
            with self.subTest(kind=kind):
                command = network_fault_arguments({"type": kind, "latency_ms": 500, "jitter_ms": 20})
                self.assertEqual(command[:5], ("delay", "500ms", "20ms", "distribution", "normal"))
