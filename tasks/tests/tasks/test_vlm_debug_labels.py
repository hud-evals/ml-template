"""Tests for the vlm_debug_labels task: bad label mask patch."""

import os

from ..conftest import apply_task_patches, copy_source_file, make_workspace


class TestVlmDebugLabels:
    def test_patch_inverts_label_mask(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        copy_source_file(ws, "torchtitan/experiments/vlm/datasets/mm_datasets.py")
        original = open(os.path.join(ws, "torchtitan/experiments/vlm/datasets/mm_datasets.py")).read()
        assert "Mask special tokens in labels" in original

        apply_task_patches(ws, "vlm_debug_labels")

        mutated = open(os.path.join(ws, "torchtitan/experiments/vlm/datasets/mm_datasets.py")).read()
        assert "torch.isin(labels, special_token_ids), labels, special_tokens.ignore_id" in mutated
