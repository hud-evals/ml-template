"""Tests for checkpoint existence grader."""

import os

from ..conftest import make_checkpoint, make_workspace, run_check


class TestCheckpointScript:
    def test_any_checkpoint_pass(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        make_checkpoint(ws, "ckpt_a", "finetune")
        assert run_check("check_checkpoint", [ws]).returncode == 0

    def test_any_checkpoint_fail(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        assert run_check("check_checkpoint", [ws]).returncode == 1

    def test_multi_ckpt_pass(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        make_checkpoint(ws, "ckpt_a", "finetune")
        make_checkpoint(ws, "ckpt_b", "pretrain")
        assert run_check("check_checkpoint", [ws, "2", "no-merged"]).returncode == 0

    def test_multi_ckpt_fail_one(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        make_checkpoint(ws, "ckpt_a", "finetune")
        assert run_check("check_checkpoint", [ws, "2", "no-merged"]).returncode == 1

    def test_dcp_checkpoint(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        ckpt = os.path.join(ws, "checkpoints", "vlm", "checkpoint", "step-100")
        os.makedirs(ckpt, exist_ok=True)
        with open(os.path.join(ckpt, ".__0_0.distcp"), "wb") as f:
            f.write(b"\x00" * 64)
        assert run_check("check_checkpoint", [ws]).returncode == 0
