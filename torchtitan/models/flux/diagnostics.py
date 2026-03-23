# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

import torch

from torchtitan.models.flux.config_registry import flux_debugmodel
from torchtitan.models.flux.trainer import FluxTrainer
from torchtitan.models.flux.utils import (
    create_position_encoding_for_latents,
    generate_noise_latent,
    pack_latents,
)


def _workspace() -> Path:
    return Path(__file__).resolve().parents[3]


def _init_dist() -> None:
    if torch.distributed.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_folder", default="checkpoints/flux")
    parser.add_argument("--checkpoint_folder", default="checkpoint")
    parser.add_argument("--load_step", type=int, default=-1)
    args = parser.parse_args()

    _init_dist()
    torch.manual_seed(12345)

    workspace = _workspace()
    tokenizers = workspace / "tests" / "assets" / "tokenizer"
    encoders = workspace / "tests" / "assets" / "flux_test_encoders"
    cc12m = workspace / "data" / "cc12m"

    config = flux_debugmodel()
    config.dump_folder = str(workspace / args.dump_folder)
    config.hf_assets_path = str(tokenizers)
    config.encoder.test_mode = True
    config.encoder.clip_encoder = str(encoders / "clip-vit-large-patch14")
    config.encoder.t5_encoder = str(encoders / "t5-v1_1-xxl")
    config.dataloader.encoder.test_mode = True
    config.dataloader.dataset_path = str(cc12m)
    config.dataloader.hf_assets_path = str(tokenizers)
    config.validator.enable = False
    config.checkpoint.folder = args.checkpoint_folder
    config.checkpoint.load_only = True
    config.checkpoint.exclude_from_loading = ["optimizer", "lr_scheduler", "dataloader"]

    trainer = FluxTrainer(config)
    try:
        loaded = trainer.checkpointer.load(step=args.load_step)
        if not loaded:
            raise FileNotFoundError(
                f"No checkpoint found under {config.dump_folder}/{config.checkpoint.folder}"
            )

        model = trainer.model_parts[0]
        model.eval()
        device = trainer.device
        dtype = trainer._dtype
        with torch.no_grad():
            bsz = 4
            noise = generate_noise_latent(bsz, 256, 256, device, dtype)
            _, _, latent_height, latent_width = noise.shape
            latent_pos = create_position_encoding_for_latents(
                bsz, latent_height, latent_width, 3
            ).to(noise)
            text_pos = torch.zeros(bsz, 256, 3, device=device, dtype=dtype)
            packed_noise = pack_latents(noise)
            timesteps = torch.full((bsz,), 0.5, dtype=dtype, device=device)
            text = torch.randn(bsz, 256, 4096, device=device, dtype=dtype)
            clip = torch.randn(bsz, 768, device=device, dtype=dtype)

            pred_cond = model(
                img=packed_noise,
                img_ids=latent_pos,
                txt=text,
                txt_ids=text_pos,
                y=clip,
                timesteps=timesteps,
            )
            pred_uncond = model(
                img=packed_noise,
                img_ids=latent_pos,
                txt=torch.zeros_like(text),
                txt_ids=text_pos,
                y=torch.zeros_like(clip),
                timesteps=timesteps,
            )

            timesteps_lo = torch.full((bsz,), 0.1, dtype=dtype, device=device)
            timesteps_hi = torch.full((bsz,), 0.9, dtype=dtype, device=device)
            pred_t_lo = model(
                img=packed_noise,
                img_ids=latent_pos,
                txt=text,
                txt_ids=text_pos,
                y=clip,
                timesteps=timesteps_lo,
            )
            pred_t_hi = model(
                img=packed_noise,
                img_ids=latent_pos,
                txt=text,
                txt_ids=text_pos,
                y=clip,
                timesteps=timesteps_hi,
            )

        diagnostics = {
            "cond_uncond_mse": ((pred_cond - pred_uncond) ** 2).mean().item(),
            "cond_uncond_cosine": torch.nn.functional.cosine_similarity(
                pred_cond.flatten(1), pred_uncond.flatten(1)
            ).mean().item(),
            "output_variance": pred_cond.var().item(),
            "timestep_cosine": torch.nn.functional.cosine_similarity(
                pred_t_lo.flatten(1), pred_t_hi.flatten(1)
            ).mean().item(),
        }
    finally:
        trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    diagnostics_path = Path(config.dump_folder) / "diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(diagnostics, sort_keys=True))


if __name__ == "__main__":
    main()
