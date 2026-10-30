"""Check pod selection and resource attribution during replacement."""

import json
import unittest
from subprocess import CompletedProcess
from unittest.mock import Mock

from et_experiments.benchmarking.sampling import Sampler, resource_summary


class SamplingTests(unittest.TestCase):
    def test_terminating_pods_cannot_replace_active_pod_observations(self):
        active = {
            "metadata": {"name": "new", "labels": {"app": "worker"}},
            "status": {"containerStatuses": [{"ready": True, "restartCount": 0}]},
        }
        terminating = {
            "metadata": {"name": "old", "labels": {"app": "worker"}, "deletionTimestamp": "2026-10-26T00:00:00Z"},
            "status": {"containerStatuses": [{"ready": False, "restartCount": 5}]},
        }
        for pods in ([active, terminating], [terminating, active], [terminating]):
            with self.subTest(names=[p["metadata"]["name"] for p in pods]):
                backend = Mock(namespace="test")
                backend.kubectl.side_effect = [
                    CompletedProcess([], 0, json.dumps({"items": pods})),
                    CompletedProcess([], 0, "old 999m 999Mi\nnew 10m 20Mi\n"),
                ]
                sampler = Sampler(backend, 1)
                sampler.stop = Mock()
                sampler.stop.is_set.side_effect = [False, True]
                sampler.start()
                sampler.thread.join(timeout=2)
                self.assertFalse(sampler.thread.is_alive())
                self.assertEqual(len(sampler.samples), 1)
                observed = sampler.samples[0]["pods"]
                if active not in pods:
                    self.assertEqual(observed, {})
                    continue
                self.assertEqual(observed["worker"]["name"], "new")
                self.assertTrue(observed["worker"]["ready"])
                self.assertEqual(observed["worker"]["restarts"], 0)
                self.assertEqual(observed["worker"]["cpu_millicores"], 10)
                self.assertEqual(observed["worker"]["memory_mib"], 20)
                summary = resource_summary(sampler.samples)["worker"]
                self.assertEqual(summary["not_ready_samples"], 0)
                self.assertEqual(summary["observed_availability_ratio"], 1)
