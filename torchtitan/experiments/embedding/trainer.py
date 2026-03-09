"""Training loop for embedding model multi-stage fine-tuning.

This module re-exports from embedding_trainer for backward compatibility.
The canonical implementation lives in embedding_trainer.py.
"""

from .embedding_trainer import EmbeddingTrainer, get_last_token_embeddings
from .train import main as train

__all__ = ["EmbeddingTrainer", "get_last_token_embeddings", "train"]
