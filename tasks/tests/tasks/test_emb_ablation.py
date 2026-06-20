"""Ablation tests: verify injected bugs cause measurable nDCG degradation.

Trains with the real scifact_finetune config and evaluates on MTEB SciFact.
If a bug doesn't cause a significant nDCG drop, the task can't distinguish
fixed from broken code and the bug needs redesigning.

Run on Modal:
    modal run modal_devbox.py --test --test-filter ablation
"""

import os
import subprocess
import sys

import pytest

# A bug must cause at least this much nDCG@10 drop to be detectable.
MIN_NDCG_GAP = 0.15


def _has_gpu():
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


gpu = pytest.mark.skipif(not _has_gpu(), reason="No GPU available")

from ..conftest import REPO_ROOT, apply_task_patches

TASKS_DIR = REPO_ROOT / "tasks"


def _setup_workspace() -> str:
    """Set up workspace identically to how the agent gets it."""
    import env

    env._setup_workspace()
    return env.WORKSPACE


def _apply_patch(ws: str, task_name: str, patch_filename: str) -> None:
    result = subprocess.run(
        ["patch", "-p1", "-i", str(TASKS_DIR / task_name / patch_filename)],
        cwd=ws, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"Patch failed: {result.stderr}"


def _apply_all_patches(ws: str, task_name: str) -> None:
    task_dir = TASKS_DIR / task_name
    for patch_file in sorted(task_dir.glob("*.patch")):
        _apply_patch(ws, task_name, patch_file.name)


def _train_and_eval_mteb(ws: str, dump_folder: str = "checkpoints/ablation") -> float:
    """Train using scifact_finetune config, return MTEB SciFact nDCG@10."""
    result = subprocess.run(
        [
            f"{ws}/.venv/bin/torchrun", "--nproc_per_node", "1",
            "-m", "torchtitan.train",
            "--module", "embedding",
            "--config", "scifact_finetune",
            "--dump_folder", dump_folder,
        ],
        cwd=ws,
        env={**os.environ, "PYTHONPATH": ws, "HOME": ws},
        capture_output=True, text=True, timeout=1200,
    )
    assert result.returncode == 0, f"Training failed:\n{result.stderr[-3000:]}"

    ckpt_dir = os.path.join(ws, dump_folder, "final")
    assert os.path.isdir(ckpt_dir), f"No checkpoint at {ckpt_dir}"

    sys.path.insert(0, ws)
    try:
        from torchtitan.experiments.embedding.evaluate import evaluate_mteb

        metrics = evaluate_mteb(ckpt_dir, ["SciFact"], max_seq_length=512)
    finally:
        sys.path.pop(0)

    ndcg = metrics.get("ndcg@10", 0.0)
    # Write to a results file so scores are retrievable regardless of pytest capture
    results_path = os.path.join(ws, "ablation_results.jsonl")
    import json as _json
    with open(results_path, "a") as f:
        f.write(_json.dumps({"checkpoint": dump_folder, "ndcg@10": ndcg}) + "\n")
    import warnings
    warnings.warn(f"ABLATION {dump_folder}: nDCG@10 = {ndcg:.4f}")
    return ndcg


@gpu
class TestBugManifests:
    """Each bug must cause >= {MIN_NDCG_GAP} nDCG@10 drop vs clean training."""

    def test_buggy_loss_manifests(self):
        ws = _setup_workspace()
        ndcg_clean = _train_and_eval_mteb(ws, "checkpoints/clean")

        _apply_patch(ws, "emb_debug_multi", "00_buggy_loss.patch")
        ndcg_buggy = _train_and_eval_mteb(ws, "checkpoints/buggy_loss")

        gap = ndcg_clean - ndcg_buggy
        print(f"  buggy_loss: clean={ndcg_clean:.4f} buggy={ndcg_buggy:.4f} gap={gap:.4f}", file=sys.stderr)
        assert gap >= MIN_NDCG_GAP, (
            f"buggy_loss doesn't manifest: clean={ndcg_clean:.4f}, "
            f"buggy={ndcg_buggy:.4f}, gap={gap:.4f} (need >={MIN_NDCG_GAP})"
        )

    def test_bad_pooling_manifests(self):
        ws = _setup_workspace()
        ndcg_clean = _train_and_eval_mteb(ws, "checkpoints/clean")

        _apply_patch(ws, "emb_debug_multi", "10_bad_pooling.patch")
        ndcg_buggy = _train_and_eval_mteb(ws, "checkpoints/buggy_pooling")

        gap = ndcg_clean - ndcg_buggy
        print(f"  bad_pooling: clean={ndcg_clean:.4f} buggy={ndcg_buggy:.4f} gap={gap:.4f}", file=sys.stderr)
        assert gap >= MIN_NDCG_GAP, (
            f"bad_pooling doesn't manifest: clean={ndcg_clean:.4f}, "
            f"buggy={ndcg_buggy:.4f}, gap={gap:.4f} (need >={MIN_NDCG_GAP})"
        )

    def test_both_bugs_manifest(self):
        ws = _setup_workspace()
        ndcg_clean = _train_and_eval_mteb(ws, "checkpoints/clean")

        _apply_all_patches(ws, "emb_debug_multi")
        ndcg_buggy = _train_and_eval_mteb(ws, "checkpoints/buggy_both")

        gap = ndcg_clean - ndcg_buggy
        print(f"  both_bugs: clean={ndcg_clean:.4f} buggy={ndcg_buggy:.4f} gap={gap:.4f}", file=sys.stderr)
        assert gap >= MIN_NDCG_GAP, (
            f"emb_debug_multi bugs don't manifest: clean={ndcg_clean:.4f}, "
            f"buggy={ndcg_buggy:.4f}, gap={gap:.4f} (need >={MIN_NDCG_GAP})"
        )
