"""Tests for the moe_seed_sweep task: bias init mutation patch."""

import os

from ..conftest import apply_task_patches, copy_source_file, make_workspace


class TestGptOssSeedSweep:
    def test_patch_removes_bias_init(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        for f in [
            "torchtitan/models/common/feed_forward.py",
            "torchtitan/models/common/moe/moe.py",
        ]:
            copy_source_file(ws, f)

        apply_task_patches(ws, "moe_seed_sweep")

        mutated_ff = open(os.path.join(ws, "torchtitan/models/common/feed_forward.py")).read()
        mutated_moe = open(os.path.join(ws, "torchtitan/models/common/moe/moe.py")).read()
        assert "nn.init.zeros_(self.w1.bias)" not in mutated_ff
        assert "nn.init.zeros_(linear.bias)" not in mutated_ff
        assert "nn.init.zeros_(self.gate.bias)" not in mutated_moe
