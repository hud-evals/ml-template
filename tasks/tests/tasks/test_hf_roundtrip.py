"""Test that last_save_in_hf checkpoints can be loaded back via initial_load_in_hf."""

import os
import subprocess

import pytest


def _has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


gpu = pytest.mark.skipif(not _has_gpu(), reason="No GPU available")


@gpu
class TestHFRoundtrip:
    def test_save_then_load(self):
        """Train with last_save_in_hf, then load that checkpoint for a second training run."""
        import env
        env._setup_workspace()
        ws = env.WORKSPACE

        # Step 1: Train 5 steps with last_save_in_hf
        result = subprocess.run(
            [
                f"{ws}/.venv/bin/torchrun", "--nproc_per_node", "1",
                "-m", "torchtitan.train",
                "--module", "embedding", "--config", "scifact_finetune",
                "--dump_folder", "checkpoints/roundtrip_step1",
                "--training.steps", "5",
                "--checkpoint.last-save-in-hf",
                "--checkpoint.last-save-model-only",
            ],
            cwd=ws,
            env={**os.environ, "PYTHONPATH": ws, "HOME": ws},
            capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, f"Step 1 failed:\n{result.stderr[-2000:]}"

        # Find the checkpoint
        ckpt_dir = os.path.join(ws, "checkpoints/roundtrip_step1/checkpoint/step-5")
        assert os.path.isdir(ckpt_dir), f"No checkpoint at {ckpt_dir}"
        files = os.listdir(ckpt_dir)
        assert any("safetensors" in f for f in files), f"No safetensors in {files}"

        # Step 2: Load that checkpoint for another training run
        result2 = subprocess.run(
            [
                f"{ws}/.venv/bin/torchrun", "--nproc_per_node", "1",
                "-m", "torchtitan.train",
                "--module", "embedding", "--config", "scifact_finetune",
                "--dump_folder", "checkpoints/roundtrip_step2",
                "--training.steps", "3",
                "--checkpoint.initial_load_path", ckpt_dir,
            ],
            cwd=ws,
            env={**os.environ, "PYTHONPATH": ws, "HOME": ws},
            capture_output=True, text=True, timeout=300,
        )

        # Report what happened
        assert False, (
            f"ROUNDTRIP step2 returncode={result2.returncode}\n"
            f"step1 checkpoint files: {files}\n"
            f"stdout (last 500): {result2.stdout[-500:]}\n"
            f"stderr (last 1000): {result2.stderr[-1000:]}"
        )
