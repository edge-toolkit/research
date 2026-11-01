import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

MODULE_PATH = Path(__file__).with_name("controller.py")
SPEC = importlib.util.spec_from_file_location("harbour_fl_orchestrator", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def node(name, *, ready=True, schedulable=True, memory=1000, cpu=1000, labels=None):
    """Build a node observation with overridable health, capacity and placement labels."""
    return MODULE.NodeState(name, ready, schedulable, labels or {}, 4000, 8192, cpu, memory)


class LeastAllocatedPolicyTests(unittest.TestCase):
    def test_experiment_and_manual_job_identities_remain_aligned(self):
        """Verify experiment and manual job identities remain aligned."""
        runtime = Mock()
        adapter = MODULE.HarbourFLAdapter(runtime)
        adapter.prepare("experiment", 3, 30)
        for call in runtime.set_env.call_args_list:
            self.assertIn("HARBOUR_FL_JOB_ID=experiment", call.args[1])
            self.assertIn("RECOVERY_JOB_ID=experiment", call.args[1])
        self.assertEqual(runtime.set_env.call_count, 4)
        runtime.reset_mock()
        with patch.object(MODULE.Path, "read_text", return_value=json.dumps({"job_id": "custom-manual-job"})):
            adapter.restore_manual_start()
        self.assertEqual(runtime.set_env.call_count, 4)
        for call in runtime.set_env.call_args_list:
            self.assertIn("HARBOUR_FL_JOB_ID=custom-manual-job", call.args[1])
            self.assertIn("RECOVERY_JOB_ID=custom-manual-job", call.args[1])
            self.assertNotIn("HARBOUR_FL_JOB_ID-", call.args[1])

    def test_restore_selector_removes_added_hostname_and_preserves_original_values(self):
        """Verify restore selector removes added hostname and preserves original values."""
        runtime = MODULE.KubernetesRuntime("test-context", "test-namespace")
        for original in (None, {"vessel-data": "true"}, {"vessel-data": "true", "kubernetes.io/hostname": "old"}):
            with self.subTest(original=original), patch.object(runtime, "_run") as run:
                runtime.restore_placement("vessel-2", original, 30)
                arguments = run.call_args_list[0].args
                selector = json.loads(arguments[arguments.index("-p") + 1])["spec"]["template"]["spec"]["nodeSelector"]
                if original is None:
                    self.assertIsNone(selector)
                else:
                    restored = {"vessel-data": "true", "kubernetes.io/hostname": "relocation-target"}
                    for key, value in selector.items():
                        if value is None:
                            restored.pop(key, None)
                        else:
                            restored[key] = value
                    self.assertEqual(restored, original)

    def test_restore_placement_propagates_patch_and_rollout_failures(self):
        """Surface nonzero kubectl exits so cleanup cannot report successful restoration."""
        runtime = MODULE.KubernetesRuntime("test-context", "test-namespace")
        for failed_step in (0, 1):
            with self.subTest(failed_step=failed_step):
                results = [subprocess.CompletedProcess([], 0, "", "")] * failed_step
                results.append(subprocess.CompletedProcess([], 1, "", "restoration rejected"))
                with (
                    patch("subprocess.run", side_effect=results) as run,
                    self.assertRaisesRegex(MODULE.OrchestrationError, "restoration rejected"),
                ):
                    runtime.restore_placement("vessel-2", {"vessel-data": "true"}, 30)
                self.assertEqual(run.call_count, failed_step + 1)
                self.assertEqual(run.call_args_list[0].args[0][3], "patch")
                if failed_step == 1:
                    self.assertEqual(run.call_args.args[0][3:5], ("rollout", "status"))
                    self.assertEqual(run.call_args.kwargs["timeout"], 40)

    def setUp(self):
        self.workload = MODULE.WorkloadState("vessel-2", "vessel-2-old", "node-a", True, 0)

    def test_selects_highest_capacity_eligible_alternate(self):
        policy = MODULE.LeastAllocatedPolicy({"vessel-data": "true"}, True)
        decision = policy.decide(
            self.workload,
            [
                node("node-a", labels={"vessel-data": "true"}),
                node("node-b", memory=3000, labels={"vessel-data": "true"}),
                node("node-c", memory=500, labels={"vessel-data": "true"}),
                node("node-d", memory=0, labels={"vessel-data": "false"}),
            ],
        )
        self.assertEqual(decision.action, "relocate")
        self.assertEqual(decision.target_node, "node-c")

    def test_ignores_unready_and_unschedulable_nodes(self):
        policy = MODULE.LeastAllocatedPolicy({}, True)
        decision = policy.decide(
            self.workload,
            [
                node("node-a"),
                node("node-b", ready=False),
                node("node-c", schedulable=False),
            ],
        )
        self.assertEqual(decision.action, "restart")
        self.assertEqual(decision.target_node, "node-a")

    def test_can_refuse_same_node_fallback(self):
        policy = MODULE.LeastAllocatedPolicy({}, False)
        with self.assertRaises(MODULE.OrchestrationError):
            policy.decide(self.workload, [node("node-a")])

    def test_resource_quantity_parsing(self):
        """Convert decimal and binary memory units, fractional quantities and bare bytes into MiB."""
        self.assertEqual(MODULE.parse_cpu("250m"), 250)
        self.assertEqual(MODULE.parse_cpu("2"), 2000)
        cases = {
            "0": 0,
            "1048576": 1,
            "1024Ki": 1,
            "2Mi": 2,
            "2Gi": 2048,
            "1.5Gi": 1536,
            "2Ti": 2 * 2**20,
            "2Pi": 2 * 2**30,
            "2Ei": 2 * 2**40,
            "512M": 488.28125,
            "1k": 1000 / 2**20,
            "1.5G": 1_500_000_000 / 2**20,
            "1T": 10**12 / 2**20,
            "1P": 10**15 / 2**20,
            "1E": 10**18 / 2**20,
        }
        for quantity, expected in cases.items():
            with self.subTest(quantity=quantity):
                self.assertAlmostEqual(MODULE.parse_memory(quantity), expected)

    def test_node_discovery_accepts_decimal_memory_requests(self):
        """Include decimal memory requests from cluster pods without interrupting placement discovery."""
        runtime = MODULE.KubernetesRuntime("test-context", "test-namespace")
        nodes = {"items": [{"metadata": {"name": "node-a"}, "status": {"allocatable": {"memory": "2Gi"}}}]}
        pods = {
            "items": [
                {
                    "spec": {
                        "nodeName": "node-a",
                        "containers": [{"resources": {"requests": {"memory": "512M"}}}],
                    }
                }
            ]
        }
        with patch.object(runtime, "_json", side_effect=[nodes, pods]):
            discovered = runtime.nodes()
        self.assertEqual(len(discovered), 1)
        self.assertEqual(discovered[0].allocatable_memory_mib, 2048)
        self.assertEqual(discovered[0].requested_memory_mib, 488.28125)

    def test_policy_validation_restricts_mutation_target(self):
        with self.assertRaises(MODULE.OrchestrationError):
            MODULE.validate_policy({"workload_id": "ws-server", "rounds": 5, "trigger_after_round": 0})
        with self.assertRaises(MODULE.OrchestrationError):
            MODULE.validate_policy({"workload_id": "vessel-2", "rounds": 3, "trigger_after_round": 2})


class SchedulingCleanupTests(unittest.TestCase):
    def setUp(self):
        """Provide an in-memory runtime and report directory for complete scheduling operations."""
        self.runtime = Mock()
        workload = MODULE.WorkloadState("vessel-2", "pod", "node-a", True, 0)
        self.runtime.workload.return_value = workload
        self.runtime.restart.return_value = workload
        self.runtime.nodes.return_value = []
        self.runtime.placement.return_value = {"vessel-data": "true"}
        self.adapter = Mock()
        self.enterContext(patch.object(MODULE, "KubernetesRuntime", return_value=self.runtime))
        clients = MODULE.HarbourFLAdapter.CLIENTS
        factory = self.enterContext(patch.object(MODULE, "HarbourFLAdapter", return_value=self.adapter))
        factory.CLIENTS = clients
        self.wait = self.enterContext(patch.object(MODULE, "wait_for_round", return_value=5))
        self.output = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_cleanup_failures_preserve_the_original_error_and_still_write_a_report(self):
        """Retain the operation error and both cleanup failures in the finalized report."""
        original = MODULE.OrchestrationError("training stopped")
        self.wait.side_effect = [1, original]
        self.runtime.restore_placement.side_effect = subprocess.TimeoutExpired("kubectl", 1)
        self.adapter.restore_manual_start.side_effect = OSError("manual restore failed")
        with self.assertLogs(level="ERROR"), self.assertRaises(MODULE.OrchestrationError) as caught:
            MODULE.execute({"action_mode": "restart"}, self.output, False)
        self.assertIs(caught.exception, original)
        report = json.loads(next(self.output.glob("*.json")).read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"], "training stopped")
        self.assertEqual([error["step"] for error in report["cleanup_errors"]], ["placement", "manual settings"])
        self.assertIn("wall_time_seconds", report)
        self.adapter.restore_manual_start.assert_called_once()

    def test_cleanup_failure_turns_a_successful_operation_into_a_failed_report(self):
        """Make cleanup failure visible to the batch loop after an otherwise successful operation."""
        self.runtime.restore_placement.side_effect = subprocess.TimeoutExpired("kubectl", 1)
        with self.assertLogs(level="ERROR"):
            report = MODULE.execute({"action_mode": "restart"}, self.output, False)
        self.assertEqual(report["status"], "failed")
        self.assertIn("Cleanup failed", report["error"])
        self.adapter.restore_manual_start.assert_called_once()
        stored = json.loads(next(self.output.glob("*.json")).read_text())
        self.assertEqual(stored, json.loads(json.dumps(report)))

    def test_a_rollout_timeout_still_attempts_placement_restoration(self):
        """Restore after a mutation that may have applied before its rollout timed out."""
        original = subprocess.TimeoutExpired("kubectl rollout", 1)
        self.runtime.restart.side_effect = original
        with self.assertRaises(subprocess.TimeoutExpired) as caught:
            MODULE.execute({"action_mode": "restart"}, self.output, False)
        self.assertIs(caught.exception, original)
        self.runtime.restore_placement.assert_called_once_with("vessel-2", {"vessel-data": "true"}, 180)
        self.adapter.restore_manual_start.assert_called_once()
        report = json.loads(next(self.output.glob("*.json")).read_text())
        self.assertEqual(report["error"], str(original))


if __name__ == "__main__":
    unittest.main()
