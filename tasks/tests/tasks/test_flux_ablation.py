"""Ablation test: timestep bug across step counts, clean vs buggy.

Run on Modal:
    modal run modal_devbox.py --test --test-filter flux_ablation
"""

import json
import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

from ..conftest import REPO_ROOT

TASKS_DIR = REPO_ROOT / "tasks"


def _has_gpu():
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


gpu = pytest.mark.skipif(not _has_gpu(), reason="No GPU available")


def _setup_workspace() -> str:
    import env

    env._setup_workspace(
        f"python {env.SRC_DIR}/tasks/utils/setup_fixtures.py "
        f"{env.WORKSPACE} --data-files pixparse/cc12m-wds "
        "cc12m-train-0000.tar cc12m-train-0001.tar cc12m-train-0002.tar"
    )
    return env.WORKSPACE


def _apply_timestep_bug(ws: str) -> None:
    for patch_file in sorted((TASKS_DIR / "flux_debug_timestep").glob("*.patch")):
        result = subprocess.run(
            ["patch", "-p1", "-i", str(patch_file)],
            cwd=ws, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Patch failed: {result.stderr}"


def _train_and_diagnose(ws: str, steps: int, dump_folder: str) -> dict:
    script = textwrap.dedent(f"""\
        import os, sys, json, torch
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

        sys.path.insert(0, "{ws}")
        from torchtitan.models.flux.config_registry import flux_debugmodel
        from torchtitan.models.flux.trainer import FluxTrainer
        from torchtitan.models.flux.utils import (
            create_position_encoding_for_latents,
            generate_noise_latent,
            pack_latents,
        )
        from pathlib import Path

        torch.distributed.init_process_group(backend="nccl")
        torch.manual_seed(12345)

        config = flux_debugmodel()
        config.dump_folder = "{ws}/{dump_folder}"
        config.hf_assets_path = "{ws}/tests/assets/tokenizer"
        config.encoder.test_mode = True
        config.encoder.clip_encoder = "{ws}/tests/assets/flux_test_encoders/clip-vit-large-patch14"
        config.encoder.t5_encoder = "{ws}/tests/assets/flux_test_encoders/t5-v1_1-xxl"
        config.dataloader.encoder.test_mode = True
        config.dataloader.dataset_path = "{ws}/data/cc12m"
        config.dataloader.hf_assets_path = "{ws}/tests/assets/tokenizer"
        config.training.steps = {steps}
        config.validator.enable = False

        trainer = FluxTrainer(config)
        try:
            trainer.train()

            model = trainer.model_parts[0]
            model.eval()
            device = trainer.device
            dtype = trainer._dtype
            with torch.no_grad():
                bsz = 4
                noise = generate_noise_latent(bsz, 256, 256, device, dtype)
                _, _, lh, lw = noise.shape
                latent_pos = create_position_encoding_for_latents(bsz, lh, lw, 3).to(noise)
                text_pos = torch.zeros(bsz, 256, 3, device=device, dtype=dtype)
                packed_noise = pack_latents(noise)
                timesteps = torch.full((bsz,), 0.5, dtype=dtype, device=device)
                text = torch.randn(bsz, 256, 4096, device=device, dtype=dtype)
                clip = torch.randn(bsz, 768, device=device, dtype=dtype)

                pred_cond = model(
                    img=packed_noise, img_ids=latent_pos,
                    txt=text, txt_ids=text_pos, y=clip, timesteps=timesteps,
                )
                pred_uncond = model(
                    img=packed_noise, img_ids=latent_pos,
                    txt=torch.zeros_like(text), txt_ids=text_pos,
                    y=torch.zeros_like(clip), timesteps=timesteps,
                )

            cosine = torch.nn.functional.cosine_similarity(
                pred_cond.flatten(1), pred_uncond.flatten(1)
            ).mean().item()

            result = {{"cond_uncond_cosine": cosine}}
            out = Path("{ws}/{dump_folder}/results.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result))
        finally:
            trainer.close()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
    """)

    script_path = os.path.join(ws, "_flux_ablation.py")
    with open(script_path, "w") as f:
        f.write(script)

    result = subprocess.run(
        [f"{ws}/.venv/bin/python", script_path],
        cwd=ws,
        env={**os.environ, "PYTHONPATH": ws, "HOME": ws},
        capture_output=True, text=True, timeout=1800,
    )
    assert result.returncode == 0, f"Train+diagnose failed:\n{result.stderr[-3000:]}"

    with open(os.path.join(ws, dump_folder, "results.json")) as f:
        return json.load(f)


STEP_COUNTS = [100, 200, 500, 1000, 2000]


def _apply_attn_bug(ws: str) -> None:
    for patch_file in sorted((TASKS_DIR / "variants" / "flux_debug_attn").glob("*.patch")):
        result = subprocess.run(
            ["patch", "-p1", "-i", str(patch_file)],
            cwd=ws, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Patch failed: {result.stderr}"


@gpu
class TestFluxTimestepAblation:

    @pytest.mark.parametrize("steps", STEP_COUNTS)
    def test_clean(self, steps):
        ws = _setup_workspace()
        metrics = _train_and_diagnose(ws, steps, f"outputs/ts_clean_{steps}")
        assert False, f"TIMESTEP CLEAN steps={steps}: cosine={metrics['cond_uncond_cosine']:.4f}"

    @pytest.mark.parametrize("steps", STEP_COUNTS)
    def test_buggy(self, steps):
        ws = _setup_workspace()
        _apply_timestep_bug(ws)
        metrics = _train_and_diagnose(ws, steps, f"outputs/ts_buggy_{steps}")
        assert False, f"TIMESTEP BUGGY steps={steps}: cosine={metrics['cond_uncond_cosine']:.4f}"


@gpu
class TestFluxAttnAblation:

    @pytest.mark.parametrize("steps", [200, 500, 1000])
    def test_clean(self, steps):
        ws = _setup_workspace()
        metrics = _train_and_diagnose(ws, steps, f"outputs/attn_clean_{steps}")
        assert False, f"ATTN CLEAN steps={steps}: cosine={metrics['cond_uncond_cosine']:.4f}"

    @pytest.mark.parametrize("steps", [200, 500, 1000])
    def test_buggy(self, steps):
        ws = _setup_workspace()
        _apply_attn_bug(ws)
        metrics = _train_and_diagnose(ws, steps, f"outputs/attn_buggy_{steps}")
        assert False, f"ATTN BUGGY steps={steps}: cosine={metrics['cond_uncond_cosine']:.4f}"
