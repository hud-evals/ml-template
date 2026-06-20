"""Tests for the emb_debug_multi task: buggy loss temperature patch."""

import os

from ..conftest import apply_task_patch, copy_source_file, make_workspace


class TestEmbDebugLoss:
    def test_patch_removes_positive_temperature_scaling(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        copy_source_file(ws, "torchtitan/experiments/embedding/losses.py")
        original = open(os.path.join(ws, "torchtitan/experiments/embedding/losses.py")).read()
        assert "pos_scores = (query_embeds * positive_embeds).sum(dim=-1) / temperature" in original

        apply_task_patch(ws, "emb_debug_multi", "00_buggy_loss.patch")

        mutated = open(os.path.join(ws, "torchtitan/experiments/embedding/losses.py")).read()
        assert "pos_scores = (query_embeds * positive_embeds).sum(dim=-1)" in mutated
        assert "pos_scores = (query_embeds * positive_embeds).sum(dim=-1) / temperature" not in mutated
