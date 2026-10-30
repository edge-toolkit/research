"""Verify subprocess deadlines without launching external commands."""

import subprocess
import unittest
from unittest.mock import patch

from et_experiments.backends.kubernetes import KubernetesBackend
from et_experiments.utils import run


class CommandTests(unittest.TestCase):
    def test_commands_use_finite_default_and_preserve_timeout_overrides(self):
        for options, expected in (({}, 600), ({"timeout": 12}, 12), ({"timeout": None}, None)):
            with self.subTest(options=options), patch("et_experiments.utils.subprocess.run") as execute:
                execute.return_value = subprocess.CompletedProcess([], 0, stdout="ok")
                self.assertEqual(run("example", **options).stdout, "ok")
                self.assertEqual(execute.call_args.kwargs["timeout"], expected)

    def test_kubernetes_uses_default_deadline_and_propagates_timeouts(self):
        backend = KubernetesBackend("ctx", "ns", "primary", ())
        error = subprocess.TimeoutExpired("kubectl", 600)
        with (
            patch("et_experiments.utils.subprocess.run", side_effect=error) as execute,
            self.assertRaises(subprocess.TimeoutExpired) as raised,
        ):
            backend.kubectl("get", "pods")
        self.assertIs(raised.exception, error)
        self.assertEqual(execute.call_args.kwargs["timeout"], 600)
