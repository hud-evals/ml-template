"""Tests for the emb_debug_loss task: buggy loss normalization patch."""

import os

from ..conftest import apply_task_patches, copy_source_file, make_workspace


class TestEmbDebugLoss:
    def test_patch_removes_normalization(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        copy_source_file(ws, "torchtitan/experiments/embedding/losses.py")
        original = open(os.path.join(ws, "torchtitan/experiments/embedding/losses.py")).read()
        assert "F.normalize" in original

        apply_task_patches(ws, "emb_debug_loss")

        mutated = open(os.path.join(ws, "torchtitan/experiments/embedding/losses.py")).read()
        assert "F.normalize(query_embeds" not in mutated
        assert "F.normalize(positive_embeds" not in mutated
        assert "F.normalize(negative_embeds" not in mutated
