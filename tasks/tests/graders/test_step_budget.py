"""Tests for the check_step_budget grader."""

import os

from ..conftest import make_workspace, run_check


def _make_training_log(ws: str, name: str, step: int) -> str:
    """Create a fake training log with a final ``step: N`` line."""
    log_path = os.path.join(ws, f"{name}.log")
    with open(log_path, "w") as f:
        f.write("step: 1\n")
        f.write(f"step: {step}\n")
    return log_path


class TestStepBudget:
    def test_passes_when_under_budget(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        _make_training_log(ws, "run_a", 80)
        _make_training_log(ws, "run_b", 90)

        result = run_check("check_step_budget", ["200", ws])
        assert result.returncode == 0, f"Should pass under budget: {result.stdout}"

    def test_fails_when_over_budget(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        _make_training_log(ws, "run_a", 120)
        _make_training_log(ws, "run_b", 95)

        result = run_check("check_step_budget", ["200", ws])
        assert result.returncode == 1, f"Should fail over budget: {result.stdout}"

    def test_uses_max_step_per_dump_folder(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        # Multiple step lines in the same log -- only the max counts.
        log_path = _make_training_log(ws, "run_a", 50)
        with open(log_path, "a") as f:
            f.write("step: 100\n")

        result = run_check("check_step_budget", ["150", ws])
        assert result.returncode == 0, f"Should use max step per log file: {result.stdout}"

    def test_no_checkpoints_passes(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        result = run_check("check_step_budget", ["200", ws])
        assert result.returncode == 0, f"No checkpoints should pass: {result.stdout}"
