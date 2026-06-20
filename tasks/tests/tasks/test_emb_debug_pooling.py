"""Tests for the emb_debug_multi task: bad pooling strategy patch."""

import os

from ..conftest import apply_task_patch, copy_source_file, make_workspace


class TestEmbDebugPooling:
    def test_patch_injects_bad_pooling(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        copy_source_file(ws, "torchtitan/experiments/embedding/embedding_trainer.py")
        original = open(os.path.join(ws, "torchtitan/experiments/embedding/embedding_trainer.py")).read()
        assert "seq_lengths = attention_mask.sum" in original

        apply_task_patch(ws, "emb_debug_multi", "10_bad_pooling.patch")

        mutated = open(os.path.join(ws, "torchtitan/experiments/embedding/embedding_trainer.py")).read()
        assert "last_hidden = hidden[:, 0, :]" in mutated
        assert "last_hidden = hidden[batch_indices, seq_lengths]" not in mutated
