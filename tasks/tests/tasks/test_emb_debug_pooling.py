"""Tests for the emb_debug_pooling task: bad pooling strategy patch."""

import os

from ..conftest import apply_task_patches, copy_source_file, make_workspace


class TestEmbDebugPooling:
    def test_patch_injects_bad_pooling(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        copy_source_file(ws, "torchtitan/experiments/embedding/embedding_trainer.py")
        original = open(os.path.join(ws, "torchtitan/experiments/embedding/embedding_trainer.py")).read()
        assert "seq_lengths = attention_mask.sum" in original

        apply_task_patches(ws, "emb_debug_pooling")

        mutated = open(os.path.join(ws, "torchtitan/experiments/embedding/embedding_trainer.py")).read()
        assert "last_hidden = (hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=-1, keepdim=True)" in mutated
