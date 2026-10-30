"""Verify bounded scheduling faults preserve desired state and restore only their own selector."""

import copy
import json
import unittest
from unittest.mock import Mock, patch

from et_experiments.benchmarking.faults import KubernetesFaults
from et_experiments.benchmarking.scheduling import SELECTOR, SELECTOR_PATH, SchedulingFaults
from et_experiments.contracts import WorkloadConfig
from et_experiments.utils import BenchmarkError


class Backend:
    namespace = "ns"

    def __init__(self):
        self.document = {"spec": {"replicas": 1, "template": {"spec": {"nodeSelector": {"disk": "ssd"}}}}}
        self.patches = []
        self.matching_nodes = []
        self.fail_add = False
        self.fail_remove = False

    def kubectl(self, *args):
        if args[:2] == ("get", "deployment/one"):
            return Mock(stdout=json.dumps(self.document))
        if args[:2] == ("get", "nodes"):
            return Mock(stdout=json.dumps({"items": self.matching_nodes}))
        if args[0] == "patch":
            operations = json.loads(args[-1])
            self.patches.append(operations)
            for operation in operations:
                if operation["op"] == "add":
                    self.document["spec"]["template"]["spec"]["nodeSelector"] = operation["value"]
                    if self.fail_add:
                        raise BenchmarkError("response lost after applying selector")
                elif operation["op"] == "remove":
                    if self.fail_remove:
                        self.fail_remove = False
                        raise BenchmarkError("temporary cleanup failure")
                    del self.document["spec"]["template"]["spec"]["nodeSelector"][SELECTOR]
            return Mock(stdout="patched")
        raise AssertionError(args)


class SchedulingTests(unittest.TestCase):
    def test_hold_preserves_one_replica_and_restores_original_placement(self):
        backend = Backend()
        original = copy.deepcopy(backend.document)
        fault = SchedulingFaults(backend, ("one",))

        def during_hold(_seconds):
            self.assertEqual(backend.document["spec"]["replicas"], 1)
            self.assertIn(SELECTOR, backend.document["spec"]["template"]["spec"]["nodeSelector"])

        with patch("et_experiments.benchmarking.scheduling.time.sleep", side_effect=during_hold) as sleep:
            result = fault.inject("one", 60, "job")
        sleep.assert_called_once_with(60)
        self.assertEqual(backend.document, original)
        self.assertEqual(result["status"], "restored")
        self.assertEqual(fault.pending, {})
        self.assertTrue(
            all(op["op"] == "test" for ops in backend.patches for op in ops if op["path"] == "/spec/replicas")
        )

    def test_cancellation_restores_selector(self):
        backend = Backend()
        with (
            patch("et_experiments.benchmarking.scheduling.time.sleep", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            SchedulingFaults(backend, ("one",)).inject("one", 60, "job")
        self.assertNotIn(SELECTOR, backend.document["spec"]["template"]["spec"]["nodeSelector"])

    def test_lost_apply_response_still_restores_selector(self):
        backend = Backend()
        backend.fail_add = True
        with self.assertRaisesRegex(BenchmarkError, "response lost"):
            SchedulingFaults(backend, ("one",)).inject("one", 60, "job")
        self.assertNotIn(SELECTOR, backend.document["spec"]["template"]["spec"]["nodeSelector"])

    def test_final_cleanup_can_retry_a_failed_release(self):
        backend = Backend()
        backend.fail_remove = True
        fault = SchedulingFaults(backend, ("one",))
        with (
            patch("et_experiments.benchmarking.scheduling.time.sleep"),
            self.assertRaisesRegex(BenchmarkError, "cleanup failure"),
        ):
            fault.inject("one", 60, "job")
        self.assertIn("one", fault.pending)
        fault.restore()
        self.assertEqual(fault.pending, {})
        self.assertNotIn(SELECTOR, backend.document["spec"]["template"]["spec"]["nodeSelector"])

    def test_unrelated_selector_changes_are_preserved_and_foreign_ownership_is_rejected(self):
        backend = Backend()
        fault = SchedulingFaults(backend, ("one",))

        def placement_change(_seconds):
            backend.document["spec"]["template"]["spec"]["nodeSelector"]["region"] = "west"

        with patch("et_experiments.benchmarking.scheduling.time.sleep", side_effect=placement_change):
            fault.inject("one", 60, "job")
        self.assertEqual(
            backend.document["spec"]["template"]["spec"]["nodeSelector"], {"disk": "ssd", "region": "west"}
        )
        backend.document["spec"]["template"]["spec"]["nodeSelector"][SELECTOR] = "someone-else"
        fault.pending["one"] = "ours"
        with self.assertRaisesRegex(BenchmarkError, "ownership changed"):
            fault.restore()
        self.assertEqual(backend.document["spec"]["template"]["spec"]["nodeSelector"][SELECTOR], "someone-else")

    def test_wrong_target_replica_count_or_matching_node_is_rejected_before_mutation(self):
        for case in ("target", "replicas", "matching-node", "already-held"):
            with self.subTest(case=case):
                backend = Backend()
                target = "primary" if case == "target" else "one"
                if case == "replicas":
                    backend.document["spec"]["replicas"] = 0
                if case == "matching-node":
                    backend.matching_nodes = [{"metadata": {"name": "node"}}]
                if case == "already-held":
                    backend.document["spec"]["template"]["spec"]["nodeSelector"][SELECTOR] = "existing"
                with self.assertRaises(BenchmarkError):
                    SchedulingFaults(backend, ("one",)).inject(target, 60, "job")
                self.assertEqual(backend.patches, [])

    def test_public_fault_dispatch_bounds_duration_and_does_not_scale_down(self):
        for requested, expected in ((-5, 1), (60, 60), (999, 300)):
            with self.subTest(requested=requested):
                backend = Backend()
                application = Mock(config=WorkloadConfig("ctx", "ns", "primary", ("one",)))
                with patch("et_experiments.benchmarking.scheduling.time.sleep") as sleep:
                    result = KubernetesFaults(backend, application).inject_fault(
                        {"type": "temporary_scheduling_outage", "target": "one", "duration_seconds": requested}, "job"
                    )
                sleep.assert_called_once_with(expected)
                self.assertEqual(result["desired_replicas"], 1)
                self.assertEqual(backend.patches[-1][-1]["path"], SELECTOR_PATH)
