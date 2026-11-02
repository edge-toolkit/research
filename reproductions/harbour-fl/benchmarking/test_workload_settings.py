"""Pretrained scenarios validate recovery policy without modifying Core."""

import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from et_experiments.utils import BenchmarkError, utc_now
from integration import HarbourFLAdapter, configure_run, parse_logs, recovery_config, restore
from workload_settings import policy_settings, resilience_settings


class WorkloadSettingsTests(unittest.TestCase):
    def test_resilience_overrides_are_bounded_and_dependencies_remain_deployment_owned(self):
        """Verify resilience overrides are bounded and dependencies remain deployment owned."""
        defaults = recovery_config()["resilience"]
        configured = resilience_settings({"recovery_resilience": {"stability_seconds": 10}}, defaults)
        self.assertEqual(configured["stability_seconds"], 10)
        self.assertEqual(defaults["stability_seconds"], 0)
        self.assertEqual(configured["dependencies"], defaults["dependencies"])
        for overrides in (
            {"stability_seconds": -1},
            {"stability_seconds": True},
            {"reconnect": "yes"},
            {"dependencies": {}},
            {"unknown": 1},
            {"backoff_multiplier": 17},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(BenchmarkError):
                resilience_settings({"recovery_resilience": overrides}, defaults)

    def test_policy_allows_configured_round_deadlines(self):
        """Verify policy allows configured round deadlines."""
        defaults = recovery_config()["policy"]
        self.assertEqual(policy_settings({}, defaults)["stall_seconds"], 240)
        reference = policy_settings({"model_profile": "pretrained", "round_timeout_seconds": 900}, defaults)
        self.assertEqual(reference["stall_seconds"], 960)
        self.assertEqual(defaults["stall_seconds"], 180)
        for invalid in (
            {"model_profile": "unknown"},
            {"recovery_policy": {"unknown": 2}},
            {"recovery_policy": {"stall_seconds": 0}},
            {"round_timeout_seconds": 0},
            {"recovery_policy": {"deadline_seconds": 1}},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(BenchmarkError):
                HarbourFLAdapter().prepare_scenario({"rounds": 3, **invalid})

    def test_all_scenarios_use_the_pretrained_model(self):
        """Verify all scenarios use the pretrained model."""
        scenarios = json.loads(Path(__file__).with_name("scenarios.json").read_text())
        self.assertEqual(scenarios["baseline"].get("model_profile", "pretrained"), "pretrained")
        self.assertEqual(scenarios["baseline"]["rounds"], 3)
        names = [name for name in scenarios if name.startswith("representative-")]
        self.assertEqual(len(names), 4)
        for name in names:
            prepared = HarbourFLAdapter().prepare_scenario(scenarios[name])
            self.assertEqual(prepared["model_profile"], "pretrained")
            self.assertEqual(prepared["local_epochs"], 2)
            self.assertEqual(prepared["round_timeout_seconds"], 900)

    def test_experiment_settings_restore_to_manual_defaults(self):
        """Verify experiment settings restore to manual defaults."""
        backend = Mock()
        settings = {"rounds": 20, "model_profile": "pretrained", "round_timeout_seconds": 900}
        with patch("integration.time.sleep"):
            configure_run(backend, Mock(), "job", settings, utc_now())
        calls = [call.args for call in backend.kubectl.call_args_list]
        self.assertEqual(sum("HARBOUR_FL_MODEL_PROFILE=pretrained" in call for call in calls), 4)
        self.assertTrue(any("RECOVERY_POLICY_STALL_SECONDS=960" in call for call in calls))
        backend.reset_mock()
        restore(backend)
        calls = [call.args for call in backend.kubectl.call_args_list]
        self.assertEqual(sum("HARBOUR_FL_MODEL_PROFILE-" in call for call in calls), 4)
        self.assertTrue(any("RECOVERY_POLICY_STALL_SECONDS=180" in call for call in calls))

    def test_explicit_round_work_timing_excludes_startup_delay(self):
        logs = {
            "vessel-coordinator": (
                "2026-10-26T00:00:00Z coordinator auto-start enabled\n"
                "2026-10-26T00:01:00Z coordinator completed round 0: {'round_work_seconds': 12.5}\n"
            )
        }
        result = parse_logs(logs, 1)
        self.assertEqual(result["round_duration_seconds"]["p95"], 12.5)

    def test_cleanup_attempts_every_workload_and_reports_all_command_failures(self):
        """Verify cleanup attempts every workload and reports all command failures."""
        for position, failure in enumerate(
            (BenchmarkError("API unavailable"), OSError("kubectl missing"), subprocess.TimeoutExpired("kubectl", 1))
        ):
            with self.subTest(position=position):
                backend = Mock()
                outcomes = [None] * 9
                outcomes[position] = failure
                backend.kubectl.side_effect = outcomes
                with self.assertRaisesRegex(BenchmarkError, "Application restoration failed"):
                    restore(backend)
                self.assertEqual(backend.kubectl.call_count, 9)
                self.assertTrue(all(call.kwargs["check"] for call in backend.kubectl.call_args_list))
                self.assertIn("deployment/vessel-coordinator", backend.kubectl.call_args.args)
        backend = Mock()
        backend.kubectl.side_effect = [BenchmarkError("first failure"), BenchmarkError("second failure"), *([None] * 7)]
        with self.assertRaises(BenchmarkError) as caught:
            restore(backend)
        self.assertIn("first failure", str(caught.exception))
        self.assertIn("second failure", str(caught.exception))
        self.assertEqual(backend.kubectl.call_count, 9)

    def test_augmented_model_can_complete_the_prediction_input_suite(self):
        original = {"rounds": 3, "adaptation": True, "evaluate_degraded_inputs": True}
        scenario = HarbourFLAdapter().prepare_scenario(original)
        self.assertTrue(scenario["adaptation"])
        self.assertTrue(scenario["evaluate_degraded_inputs"])
        self.assertEqual(scenario["completion_event"], "completed")
        self.assertNotIn("completion_event", original)

    def test_pretrained_report_requires_observed_workload_settings_from_every_role(self):
        """Verify pretrained report requires observed workload settings from every role."""
        adapter = HarbourFLAdapter()
        logs = {"vessel-coordinator": "coordinator completed round 0: {'global_test_accuracy': 0.5}"}
        observed = adapter.observe(logs, None, 1, utc_now())
        for complete in (False, True):
            with self.subTest(complete=complete):
                report = {
                    "status": "passed",
                    "effective_scenario": {"model_profile": "pretrained"},
                    "reference": {"conditions": {}, "missing": []},
                    "workload": observed,
                    "fault": {"type": "none"},
                    "events": [],
                    "resilience": {"completion_ratio": 1, "completed_units_per_minute": 1},
                    "diagnostic_events": [
                        {"kind": "workload-profile", "workload": name, "settings": {"model_profile": "pretrained"}}
                        for name in adapter.config.workloads
                    ]
                    if complete
                    else [],
                }
                adapter.enrich_report(report)
                self.assertEqual(report["status"], "passed" if complete else "failed")
                if complete:
                    self.assertEqual(len(report["observed_workloads"]), 4)
                    self.assertEqual(len(report["reference"]["fingerprint"]), 64)
                else:
                    self.assertIn("observed-workloads", report["reference"]["missing"])
