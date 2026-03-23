"""GPU smoke test for the VLM training pipeline."""

import os
import shutil
import subprocess

from ..conftest import REPO_ROOT, gpu


@gpu
class TestVlmFinetune:
    def test_training_runs(self, tmp_path):
        ws = str(tmp_path / "workspace")
        os.makedirs(f"{ws}/data/cc12m", exist_ok=True)

        shutil.copytree(
            str(REPO_ROOT / "tests/assets/cc12m_test"),
            f"{ws}/data/cc12m",
            dirs_exist_ok=True,
        )
        shutil.copytree(
            str(REPO_ROOT / "tests/assets/tokenizer"),
            f"{ws}/tokenizer",
        )

        env = {**os.environ, "PYTHONPATH": ws}
        shutil.copytree(str(REPO_ROOT / "torchtitan"), f"{ws}/torchtitan")

        result = subprocess.run(
            [
                "torchrun", "--nproc_per_node", "1", "-m", "torchtitan.train",
                "--module", "vlm", "--config", "vlm_debugmodel",
                "--dump_folder", f"{ws}/checkpoints/vlm",
                "--hf_assets_path", f"{ws}/tokenizer",
                "--dataloader.dataset_path", f"{ws}/data/cc12m",
                "--training.steps", "5",
                "--training.local_batch_size", "2",
            ],
            cwd=ws,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, f"VLM training failed:\n{result.stderr[-2000:]}"
