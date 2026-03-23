"""EmbeddingTrainer -- torchtitan Trainer extension for contrastive embedding training.

Uses torchtitan's Qwen3Model backbone with InfoNCE loss. Exports checkpoints
in HF format for MTEB evaluation compatibility.

Usage:
    torchrun --nproc_per_node 1 -m torchtitan.train \
        --module embedding --config scifact_finetune
"""

import logging
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.trainer import Trainer

from .configs import EmbeddingConfig
from .datasets import EmbeddingDataLoader
from .losses import infonce_loss, matryoshka_loss

logger = logging.getLogger(__name__)


def get_last_token_embeddings(hidden, attention_mask):
    """Extract embeddings from the last non-padding token (EOS position)."""
    seq_lengths = attention_mask.sum(dim=-1) - 1
    batch_indices = torch.arange(hidden.size(0), device=hidden.device)
    last_hidden = hidden[batch_indices, seq_lengths]

    return F.normalize(last_hidden, dim=-1)


class EmbeddingTrainer(Trainer):
    """Contrastive embedding trainer using torchtitan's Qwen3Model.

    Overrides ``forward_backward_step`` for triple forward pass (query/pos/neg)
    and InfoNCE loss.  The base ``Trainer.train()`` loop drives training;
    ``EmbeddingDataLoader`` handles multi-epoch iteration and the loop
    terminates on ``DataloaderExhaustedError``.

    On ``close()``, the final model is exported in HF format for evaluation.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
        tokenizer: BaseTokenizer.Config | None = None
        dataloader: EmbeddingDataLoader.Config = field(
            default_factory=EmbeddingDataLoader.Config
        )
        checkpoint: CheckpointManager.Config = field(
            default_factory=lambda: CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                last_save_in_hf=True,
                last_save_model_only=True,
                interval=999999,
            )
        )

    def __init__(self, config: Config):
        super().__init__(config)
        self.emb_config = config.embedding

    def train(self):
        """Override train to save checkpoint when data is exhausted."""
        super().train()
        # The base train() breaks on DataloaderExhaustedError without saving.
        # Force a final save so HF export happens.
        if self.step > 0 and self.step != self.config.training.steps:
            self.checkpointer.save(self.step, last_step=True)

    def _get_hidden_states(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward through model backbone, returning hidden states before output projection."""
        model = self.model_parts[0]
        h = model.tok_embeddings(tokens) if model.tok_embeddings is not None else tokens
        for layer in model.layers.values():
            h = layer(h, model.freqs_cis, None)
        h = model.norm(h) if model.norm is not None else h
        return h

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.emb_config

        query_ids = input_dict["input"]
        query_mask = input_dict["query_attention_mask"]
        pos_ids = input_dict["pos_input_ids"]
        pos_mask = input_dict["pos_attention_mask"]
        neg_ids = input_dict["neg_input_ids"]
        neg_mask = input_dict["neg_attention_mask"]

        B, K, S = neg_ids.shape

        with self.train_context():
            with self.maybe_enable_amp:
                query_hidden = self._get_hidden_states(query_ids)
                query_emb = get_last_token_embeddings(query_hidden, query_mask)

                pos_hidden = self._get_hidden_states(pos_ids)
                pos_emb = get_last_token_embeddings(pos_hidden, pos_mask)

                neg_hidden = self._get_hidden_states(neg_ids.view(B * K, S))
                neg_emb = get_last_token_embeddings(neg_hidden, neg_mask.view(B * K, S))
                neg_emb = neg_emb.view(B, K, -1)

                if emb.matryoshka_dims and len(emb.matryoshka_dims) > 0:
                    loss = matryoshka_loss(
                        query_emb,
                        pos_emb,
                        neg_emb,
                        dims=emb.matryoshka_dims,
                        temperature=emb.temperature,
                        false_neg_threshold=emb.false_negative_threshold,
                    )
                else:
                    loss = infonce_loss(
                        query_emb,
                        pos_emb,
                        neg_emb,
                        temperature=emb.temperature,
                        false_neg_threshold=emb.false_negative_threshold,
                    )

            del query_hidden, pos_hidden, neg_hidden
            loss.backward()

        return loss.detach()

