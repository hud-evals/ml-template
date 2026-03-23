"""Embedding training experiment for torchtitan.

Trains contrastive embedding models using InfoNCE loss with last-token (EOS) pooling.
Uses torchtitan's Qwen3Model backbone with HF-format checkpoint export.

Usage:
    torchrun --nproc_per_node 1 -m torchtitan.train \
        --module embedding --config scifact_finetune
"""

from torchtitan.models.qwen3 import parallelize_qwen3, qwen3_configs
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from .embedding_trainer import EmbeddingTrainer, get_last_token_embeddings
from .losses import build_infonce_loss


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="embedding",
        flavor=flavor,
        model=qwen3_configs[flavor],
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=None,
        build_loss_fn=build_infonce_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen3StateDictAdapter,
    )


__all__ = [
    "EmbeddingTrainer",
    "get_last_token_embeddings",
    "model_registry",
]
