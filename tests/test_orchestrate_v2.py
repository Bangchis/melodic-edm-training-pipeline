from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from scripts import orchestrate_v2


class SupervisorStateTests(unittest.TestCase):
    @staticmethod
    def completed(returncode: int, stdout: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["supervisorctl", "status", "job"],
            returncode=returncode,
            stdout=stdout,
            stderr="",
        )

    def test_state_accepts_supervisor_exit_code_three_for_exited_job(self) -> None:
        result = self.completed(3, "edm-v2-moss-0 EXITED Jul 18 04:55 AM\n")
        with patch.object(orchestrate_v2.subprocess, "run", return_value=result):
            self.assertEqual(orchestrate_v2.state("edm-v2-moss-0"), "EXITED")

    def test_state_accepts_supervisor_exit_code_three_for_stopped_job(self) -> None:
        result = self.completed(3, "edm-v2-build-annotations STOPPED Not started\n")
        with patch.object(orchestrate_v2.subprocess, "run", return_value=result):
            self.assertEqual(orchestrate_v2.state("edm-v2-build-annotations"), "STOPPED")

    def test_non_status_commands_still_reject_exit_code_three(self) -> None:
        result = self.completed(3, "edm-v2-build-annotations ERROR (spawn error)\n")
        with patch.object(orchestrate_v2.subprocess, "run", return_value=result):
            with self.assertRaises(RuntimeError):
                orchestrate_v2.supervisor("start", "edm-v2-build-annotations")

    def test_state_rejects_unknown_process_exit_code_four(self) -> None:
        result = self.completed(4, "edm-v2-missing ERROR (no such process)\n")
        with patch.object(orchestrate_v2.subprocess, "run", return_value=result):
            with self.assertRaises(RuntimeError):
                orchestrate_v2.state("edm-v2-missing")


if __name__ == "__main__":
    unittest.main()
