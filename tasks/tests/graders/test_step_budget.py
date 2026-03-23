"""Tests for the check_step_budget grader."""

import os

from ..conftest import make_workspace, run_check


def _make_dcp_checkpoint(ws: str, dump_folder: str, step: int) -> str:
    """Create a fake DCP checkpoint at dump_folder/step-N/."""
    ckpt_dir = os.path.join(ws, dump_folder, f"step-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, ".metadata"), "w") as f:
        f.write("{}")
    return ckpt_dir


class TestStepBudget:
    def test_passes_when_under_budget(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        _make_dcp_checkpoint(ws, "outputs/run_a/checkpoint", 80)
        _make_dcp_checkpoint(ws, "outputs/run_b/checkpoint", 90)

        result = run_check("check_step_budget", ["200", ws])
        assert result.returncode == 0, f"Should pass under budget: {result.stdout}"

    def test_fails_when_over_budget(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        _make_dcp_checkpoint(ws, "outputs/run_a/checkpoint", 120)
        _make_dcp_checkpoint(ws, "outputs/run_b/checkpoint", 95)

        result = run_check("check_step_budget", ["200", ws])
        assert result.returncode == 1, f"Should fail over budget: {result.stdout}"

    def test_uses_max_step_per_dump_folder(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        # Two checkpoints in the same dump folder -- only the max counts
        _make_dcp_checkpoint(ws, "outputs/run_a/checkpoint", 50)
        _make_dcp_checkpoint(ws, "outputs/run_a/checkpoint", 100)

        result = run_check("check_step_budget", ["150", ws])
        assert result.returncode == 0, f"Should use max step per dump folder: {result.stdout}"

    def test_no_checkpoints_passes(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        result = run_check("check_step_budget", ["200", ws])
        assert result.returncode == 0, f"No checkpoints should pass: {result.stdout}"
