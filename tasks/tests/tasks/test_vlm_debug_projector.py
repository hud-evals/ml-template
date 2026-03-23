"""Tests for the vlm_debug_projector task: buggy projector patch."""

import os

from ..conftest import apply_task_patches, copy_source_file, make_workspace


class TestVlmDebugProjector:
    def test_patch_breaks_projector(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        copy_source_file(ws, "torchtitan/experiments/vlm/model/model.py")
        original = open(os.path.join(ws, "torchtitan/experiments/vlm/model/model.py")).read()
        assert "nn.functional.silu" in original

        apply_task_patches(ws, "vlm_debug_projector")

        mutated = open(os.path.join(ws, "torchtitan/experiments/vlm/model/model.py")).read()
        assert "x_NLD = self.w2(self.w1(x_NLD[:, :1, :]))" in mutated
        assert "x_NLD = x_NLD.expand(-1, seq_len, -1)" in mutated
        assert "nn.functional.silu" not in mutated
