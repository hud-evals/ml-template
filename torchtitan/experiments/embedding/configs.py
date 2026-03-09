"""Configuration dataclasses for the embedding experiment."""

from dataclasses import dataclass
from typing import Literal


@dataclass(kw_only=True, slots=True)
class EmbeddingConfig:
    """Embedding-specific training configuration."""

    stage: Literal["pretrain", "finetune"] = "finetune"
    model_name: str = "Qwen/Qwen3-0.6B"
    resume_from: str | None = None
    temperature: float = 0.02
    num_hard_negatives: int = 7
    false_negative_threshold: float = 0.1
    matryoshka_dims: list[int] | None = None
    output_dim: int | None = None
    num_epochs: int = 3
    save_steps: int | None = None
    train_data: str = ""
    eval_data: str | None = None
