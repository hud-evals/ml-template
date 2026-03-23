"""GPU smoke test for the embedding finetune training pipeline."""

import json
import os
import subprocess

from ..conftest import REPO_ROOT, gpu, make_training_data


@gpu
class TestEmbFinetune:
    def test_training_produces_checkpoint(self, tmp_path):
        ws = str(tmp_path / "workspace")
        os.makedirs(f"{ws}/data", exist_ok=True)
        make_training_data(ws, n=20)

        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        result = subprocess.run(
            [
                "torchrun",
                "--nproc_per_node",
                "1",
                "-m",
                "torchtitan.train",
                "--module",
                "embedding",
                "--config",
                "scifact_finetune",
                "--dump_folder",
                f"{ws}/checkpoints/stage1",
                "--dataloader.num_epochs",
                "1",
                "--training.local_batch_size",
                "2",
                "--training.seq_len",
                "64",
                "--embedding.num_hard_negatives",
                "2",
                "--dataloader.num_hard_negatives",
                "2",
                "--embedding.train_data",
                f"{ws}/data/scifact.jsonl",
                "--dataloader.train_path",
                f"{ws}/data/scifact.jsonl",
            ],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, f"Training failed:\n{result.stderr[-2000:]}"

        ckpts = list(
            (tmp_path / "workspace" / "checkpoints" / "stage1").glob(
                "**/model.safetensors"
            )
        )
        assert len(ckpts) > 0, "No checkpoint produced"

        meta_files = list(
            (tmp_path / "workspace" / "checkpoints" / "stage1").glob(
                "**/training_metadata.json"
            )
        )
        assert len(meta_files) > 0, "No training metadata produced"
        meta = json.loads(meta_files[0].read_text())
        assert meta["stage"] == "finetune"
