"""EmbeddingTrainer -- torchtitan Trainer extension for contrastive embedding training.

Uses torchtitan's Qwen3Model backbone with InfoNCE loss. Exports checkpoints
in HF format for MTEB evaluation compatibility.

Invocable via:
    torchrun --nproc_per_node 1 -m torchtitan.train \
        --module embedding --config scifact_finetune
Or standalone:
    python -m torchtitan.experiments.embedding.train \
        --stage finetune --train_data data/scifact.jsonl --output_dir checkpoints
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.dataloader import DataloaderExhaustedError
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

    Overrides forward_backward_step for triple forward pass (query/pos/neg)
    and InfoNCE loss. Saves checkpoints in HF format for evaluation.
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
                interval=999999,
            )
        )

    def __init__(self, config: Config):
        super().__init__(config)
        self.emb_config = config.embedding

        from transformers import AutoTokenizer

        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            config.embedding.model_name, trust_remote_code=True
        )

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
                neg_emb = get_last_token_embeddings(
                    neg_hidden, neg_mask.view(B * K, S)
                )
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

    def _export_hf_checkpoint(
        self, name: str, avg_loss: float | None = None
    ) -> str:
        """Export current model weights in HF format for evaluation."""
        output_dir = self.config.dump_folder
        ckpt_dir = os.path.join(output_dir, name)

        if torch.distributed.get_rank() != 0:
            return ckpt_dir

        os.makedirs(ckpt_dir, exist_ok=True)

        state_dict = get_model_state_dict(self.model_parts[0])
        state_dict = {k: v.detach().cpu() for k, v in state_dict.items()}

        from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter

        adapter = Qwen3StateDictAdapter(self.model_config, None)
        hf_state_dict = adapter.to_hf(state_dict)

        from safetensors.torch import save_file

        save_file(hf_state_dict, os.path.join(ckpt_dir, "model.safetensors"))

        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(
            self.emb_config.model_name, trust_remote_code=True
        )
        hf_config.save_pretrained(ckpt_dir)
        self.hf_tokenizer.save_pretrained(ckpt_dir)

        metadata = {
            "stage": self.emb_config.stage,
            "model_name": self.emb_config.model_name,
            "resume_from": self.emb_config.resume_from,
            "checkpoint": name,
            "global_step": self.step,
            "avg_loss": avg_loss,
            "learning_rate": self.config.optimizer.lr,
            "batch_size": self.config.training.local_batch_size,
            "max_seq_length": self.config.training.seq_len,
            "temperature": self.emb_config.temperature,
            "num_hard_negatives": self.emb_config.num_hard_negatives,
            "matryoshka_dims": self.emb_config.matryoshka_dims,
        }
        with open(os.path.join(ckpt_dir, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Exported HF checkpoint: %s", ckpt_dir)
        return ckpt_dir

    def train(self):
        """Epoch-based training with HF checkpoint export."""
        config = self.config
        emb = self.emb_config

        self.checkpointer.load(step=config.checkpoint.load_step)
        logger.info("Training starts at step %d", self.step + 1)

        for epoch in range(emb.num_epochs):
            data_iterator = self.batch_generator(self.dataloader)
            epoch_steps = 0

            while True:
                self.step += 1
                self.gc_handler.run(self.step)
                try:
                    self.train_step(data_iterator)
                except DataloaderExhaustedError:
                    break
                epoch_steps += 1

            logger.info(
                "Epoch %d/%d complete (%d steps)",
                epoch + 1,
                emb.num_epochs,
                epoch_steps,
            )

            ckpt_dir = self._export_hf_checkpoint(f"epoch_{epoch + 1}")

            if emb.eval_data:
                from .evaluate import evaluate_local

                metrics = evaluate_local(
                    ckpt_dir, emb.eval_data, config.training.seq_len
                )
                logger.info("Eval metrics: %s", json.dumps(metrics, indent=2))
                with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=2)

        if torch.distributed.get_rank() == 0:
            time.sleep(2)

        logger.info("Training completed")

    def close(self):
        if hasattr(self, "checkpointer"):
            self.checkpointer.close()
        if hasattr(self, "metrics_processor"):
            self.metrics_processor.close()
