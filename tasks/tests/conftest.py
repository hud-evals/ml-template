"""Shared test helpers for grader and task tests."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GRADERS_DIR = REPO_ROOT / "tasks" / "graders"


def run_check(name: str, args: list[str] | None = None) -> subprocess.CompletedProcess:
    """Execute a grader script from tasks/graders/ with the given args."""
    script = GRADERS_DIR / f"{name}.py"
    cmd = [sys.executable, str(script)] + (args or [])
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30)


def make_workspace(tmp_path: str) -> str:
    ws = os.path.join(tmp_path, "workspace")
    os.makedirs(os.path.join(ws, "data"), exist_ok=True)
    return ws


def make_checkpoint(ws: str, name: str, stage: str, extra_meta: dict | None = None) -> str:
    ckpt_dir = os.path.join(ws, "checkpoints", name)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "model.safetensors"), "wb") as f:
        f.write(b"\x00" * 64)
    meta = {"stage": stage, "model_name": "test", **(extra_meta or {})}
    with open(os.path.join(ckpt_dir, "training_metadata.json"), "w") as f:
        json.dump(meta, f)
    return ckpt_dir


def copy_source_file(ws: str, relative_path: str) -> None:
    src = str(REPO_ROOT / relative_path)
    dst = os.path.join(ws, relative_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def apply_task_patches(ws: str, task_name: str) -> None:
    task_dir = REPO_ROOT / "tasks" / task_name
    for patch_file in sorted(os.listdir(task_dir)):
        if patch_file.endswith(".patch"):
            apply_task_patch(ws, task_name, patch_file)


def apply_task_patch(ws: str, task_name: str, patch_file: str) -> None:
    task_dir = REPO_ROOT / "tasks" / task_name
    result = subprocess.run(
        ["patch", "-p1", "-i", str(task_dir / patch_file)],
        cwd=ws, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"patch failed: {result.stderr}"


def make_training_data(ws: str, filename: str = "scifact.jsonl", n: int = 50) -> str:
    path = os.path.join(ws, "data", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for i in range(n):
            pair = {
                "instruction": "test",
                "query": f"query {i}",
                "positive": f"positive text {i}",
                "negatives": [f"negative {i}-{j}" for j in range(3)],
            }
            f.write(json.dumps(pair) + "\n")
    return path


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


gpu = pytest.mark.skipif(not _has_gpu(), reason="No GPU available")
