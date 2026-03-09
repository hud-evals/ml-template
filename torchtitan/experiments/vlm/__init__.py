# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields
from typing import Any

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.models.common import Embedding, FeedForward, GQAttention, RoPE
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.qwen3 import Qwen3TransformerBlock, qwen3_configs
from torchtitan.protocols.model_spec import ModelSpec

from .datasets.mm_datasets import HuggingFaceMultiModalDataLoader
from .infra.parallelize import parallelize_vlm
from .model.args import Siglip2Config
from .model.model import Llama3Siglip2Transformer, Qwen3Siglip2Transformer

__all__ = [
    "HuggingFaceMultiModalDataLoader",
    "parallelize_vlm",
    "Llama3Siglip2Transformer",
    "Qwen3Siglip2Transformer",
    "llama3_siglip2_configs",
    "qwen3_siglip2_configs",
]


def _get_dict(obj) -> dict[str, Any]:
    """Convert dataclass to dict, preserving nested dataclasses (unlike asdict)."""
    return {field.name: getattr(obj, field.name) for field in fields(obj)}


# Real SigLIP2 encoder sizes
SIGLIP2_DEBUG = Siglip2Config(dim=128, ffn_dim=256, n_layers=4, n_heads=2)
SIGLIP2_BASE = Siglip2Config(dim=768, ffn_dim=3072, n_layers=12, n_heads=12)
SIGLIP2_LARGE = Siglip2Config(dim=1024, ffn_dim=4096, n_layers=24, n_heads=16)


llama3_siglip2_configs = {
    "debugmodel": Llama3Siglip2Transformer.Config(
        **_get_dict(llama3_configs["debugmodel_flex_attn"]),
        encoder=SIGLIP2_DEBUG,
    ),
}


def _qwen3_flex_config(base: str, encoder: Siglip2Config) -> Qwen3Siglip2Transformer.Config:
    """Create a Qwen3+SigLIP2 VLM config with flex attention from a base Qwen3 config."""
    src = qwen3_configs[base]
    d = _get_dict(src)
    # Swap attention backend to flex for VLM encoder-decoder masking
    old_attn = d["layer"].attention
    d["layer"] = Qwen3TransformerBlock.Config(
        attention_norm=d["layer"].attention_norm,
        ffn_norm=d["layer"].ffn_norm,
        feed_forward=d["layer"].feed_forward,
        moe_enabled=d["layer"].moe_enabled,
        moe=d["layer"].moe,
        attention=GQAttention.Config(
            n_heads=old_attn.n_heads,
            n_kv_heads=old_attn.n_kv_heads,
            head_dim=old_attn.head_dim,
            q_norm=old_attn.q_norm,
            k_norm=old_attn.k_norm,
            attn_backend="flex",
            attn_mask_type="block_causal",
            rope_backend=old_attn.rope_backend,
        ),
    )
    return Qwen3Siglip2Transformer.Config(**d, encoder=encoder)


qwen3_siglip2_configs = {
    "debugmodel": _qwen3_flex_config("debugmodel", SIGLIP2_DEBUG),
    "0.6B": _qwen3_flex_config("0.6B", SIGLIP2_BASE),
    "1.7B": _qwen3_flex_config("1.7B", SIGLIP2_LARGE),
}

# Unified registry: "qwen3_0.6B", "qwen3_debugmodel", "llama3_debugmodel", "debugmodel" (legacy)
_ALL_CONFIGS = {}
for name, cfg in llama3_siglip2_configs.items():
    _ALL_CONFIGS[f"llama3_{name}"] = ("llama3", cfg)
for name, cfg in qwen3_siglip2_configs.items():
    _ALL_CONFIGS[f"qwen3_{name}"] = ("qwen3", cfg)
# Legacy alias
_ALL_CONFIGS["debugmodel"] = ("llama3", llama3_siglip2_configs["debugmodel"])


def model_registry(flavor: str) -> ModelSpec:
    if flavor not in _ALL_CONFIGS:
        available = ", ".join(sorted(_ALL_CONFIGS.keys()))
        raise ValueError(f"Unknown VLM flavor '{flavor}'. Available: {available}")
    family, config = _ALL_CONFIGS[flavor]
    return ModelSpec(
        name="vlm",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_vlm,
        pipelining_fn=None,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
