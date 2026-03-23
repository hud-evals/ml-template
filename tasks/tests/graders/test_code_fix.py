"""Tests for the check_code_fix grader."""

import subprocess

from ..conftest import REPO_ROOT, apply_task_patches, copy_source_file, make_workspace, run_check


class TestCodeFixChecks:
    def _apply_patch_file(self, ws: str, task_name: str, patch_name: str):
        patch_path = REPO_ROOT / "tasks" / task_name / patch_name
        result = subprocess.run(
            ["patch", "-p1", "-i", str(patch_path)],
            cwd=ws,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"patch failed: {result.stderr}"

    def test_buggy_loss_unfixed_fails(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        copy_source_file(ws, "torchtitan/experiments/embedding/losses.py")
        apply_task_patches(ws, "emb_debug_loss")
        r = run_check("check_code_fix", ["buggy_loss", ws])
        assert r.returncode == 1, f"Should fail on unfixed code: {r.stdout}"

    def test_buggy_loss_fixed_passes(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        copy_source_file(ws, "torchtitan/experiments/embedding/losses.py")
        r = run_check("check_code_fix", ["buggy_loss", ws])
        assert r.returncode == 0, f"Should pass on original code: {r.stdout}"

    def test_bad_pooling_unfixed_fails(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        copy_source_file(ws, "torchtitan/experiments/embedding/embedding_trainer.py")
        apply_task_patches(ws, "emb_debug_pooling")
        r = run_check("check_code_fix", ["bad_pooling", ws])
        assert r.returncode == 1, f"Should fail on unfixed code: {r.stdout}"

    def test_bad_pooling_fixed_passes(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        copy_source_file(ws, "torchtitan/experiments/embedding/embedding_trainer.py")
        r = run_check("check_code_fix", ["bad_pooling", ws])
        assert r.returncode == 0, f"Should pass on original code: {r.stdout}"

    def test_buggy_projector_unfixed_fails(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        copy_source_file(ws, "torchtitan/experiments/vlm/model/model.py")
        apply_task_patches(ws, "vlm_debug_projector")
        r = run_check("check_code_fix", ["buggy_projector", ws])
        assert r.returncode == 1, f"Should fail on unfixed code: {r.stdout}"

    def test_buggy_projector_fixed_passes(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        copy_source_file(ws, "torchtitan/experiments/vlm/model/model.py")
        r = run_check("check_code_fix", ["buggy_projector", ws])
        assert r.returncode == 0, f"Should pass on original code: {r.stdout}"

    def test_bad_label_mask_unfixed_fails(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        copy_source_file(ws, "torchtitan/experiments/vlm/datasets/mm_datasets.py")
        apply_task_patches(ws, "vlm_debug_labels")
        r = run_check("check_code_fix", ["bad_label_mask", ws])
        assert r.returncode == 1, f"Should fail on unfixed code: {r.stdout}"

    def test_bad_label_mask_fixed_passes(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        copy_source_file(ws, "torchtitan/experiments/vlm/datasets/mm_datasets.py")
        r = run_check("check_code_fix", ["bad_label_mask", ws])
        assert r.returncode == 0, f"Should pass on original code: {r.stdout}"

    def test_flux_causal_attn_unfixed_fails(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        copy_source_file(ws, "torchtitan/models/flux/model/layers.py")
        apply_task_patches(ws, "flux_debug_attn")
        r = run_check("check_code_fix", ["flux_causal_attn", ws])
        assert r.returncode == 1, f"Should fail on unfixed code: {r.stdout}"

    def test_flux_causal_attn_fixed_passes(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        copy_source_file(ws, "torchtitan/models/flux/model/layers.py")
        self._apply_patch_file(ws, "flux_debug_attn", "00_baseline.patch")
        r = run_check("check_code_fix", ["flux_causal_attn", ws])
        assert r.returncode == 0, f"Should pass on baseline-fixed code: {r.stdout}"

    def test_flux_cfg_dropout_unfixed_fails(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        copy_source_file(ws, "torchtitan/models/flux/model/layers.py")
        copy_source_file(ws, "torchtitan/models/flux/flux_datasets.py")
        apply_task_patches(ws, "flux_debug_guidance")
        r = run_check("check_code_fix", ["flux_cfg_dropout", ws])
        assert r.returncode == 1, f"Should fail on unfixed code: {r.stdout}"

    def test_flux_cfg_dropout_fixed_passes(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        copy_source_file(ws, "torchtitan/models/flux/flux_datasets.py")
        r = run_check("check_code_fix", ["flux_cfg_dropout", ws])
        assert r.returncode == 0, f"Should pass on original code: {r.stdout}"

    def test_flux_zero_timestep_unfixed_fails(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        copy_source_file(ws, "torchtitan/models/flux/model/layers.py")
        copy_source_file(ws, "torchtitan/models/flux/model/model.py")
        apply_task_patches(ws, "flux_debug_timestep")
        r = run_check("check_code_fix", ["flux_zero_timestep", ws])
        assert r.returncode == 1, f"Should fail on unfixed code: {r.stdout}"

    def test_flux_zero_timestep_fixed_passes(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        copy_source_file(ws, "torchtitan/models/flux/model/model.py")
        r = run_check("check_code_fix", ["flux_zero_timestep", ws])
        assert r.returncode == 0, f"Should pass on original code: {r.stdout}"
