"""Named configurations for the embedding experiment.

Usage via torchtitan's ConfigManager:
    torchrun --nproc_per_node 1 -m torchtitan.train \
        --module embedding --config scifact_finetune
"""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config.configs import TrainingConfig

from . import model_registry
from .configs import EmbeddingConfig
from .datasets import EmbeddingDataLoader
from .embedding_trainer import EmbeddingTrainer

_HF_ASSETS = "./assets/hf/Qwen3-0.6B"


def scifact_pretrain() -> EmbeddingTrainer.Config:
    return EmbeddingTrainer.Config(
        model_spec=model_registry("0.6B"),
        hf_assets_path=_HF_ASSETS,
        dump_folder="./checkpoints/pretrain",
        embedding=EmbeddingConfig(
            stage="pretrain",
            model_name=_HF_ASSETS,
            temperature=0.02,
            num_hard_negatives=7,
            train_data="data/synthetic.jsonl",
        ),
        dataloader=EmbeddingDataLoader.Config(
            train_path="data/synthetic.jsonl",
            model_name=_HF_ASSETS,
            num_hard_negatives=7,
            num_epochs=1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            global_batch_size=4,
            seq_len=512,
            max_norm=1.0,
            steps=999999,
        ),
        optimizer=OptimizersContainer.Config(
            lr=1e-5,
            weight_decay=0.01,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=50,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            interval=999999,
        ),
    )


def scifact_finetune() -> EmbeddingTrainer.Config:
    return EmbeddingTrainer.Config(
        model_spec=model_registry("0.6B"),
        hf_assets_path=_HF_ASSETS,
        dump_folder="./checkpoints/finetune",
        embedding=EmbeddingConfig(
            stage="finetune",
            model_name=_HF_ASSETS,
            temperature=0.02,
            num_hard_negatives=7,
            train_data="data/scifact.jsonl",
        ),
        dataloader=EmbeddingDataLoader.Config(
            train_path="data/scifact.jsonl",
            model_name=_HF_ASSETS,
            num_hard_negatives=7,
            num_epochs=3,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            global_batch_size=4,
            seq_len=512,
            max_norm=1.0,
            steps=999999,
        ),
        optimizer=OptimizersContainer.Config(
            lr=1e-5,
            weight_decay=0.01,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=50,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            interval=999999,
        ),
    )


def scifact_matryoshka() -> EmbeddingTrainer.Config:
    return EmbeddingTrainer.Config(
        model_spec=model_registry("0.6B"),
        hf_assets_path=_HF_ASSETS,
        dump_folder="./checkpoints/matryoshka",
        embedding=EmbeddingConfig(
            stage="finetune",
            model_name=_HF_ASSETS,
            temperature=0.02,
            num_hard_negatives=7,
            matryoshka_dims=[64, 128, 256, 512, 1024],
            train_data="data/scifact.jsonl",
        ),
        dataloader=EmbeddingDataLoader.Config(
            train_path="data/scifact.jsonl",
            model_name=_HF_ASSETS,
            num_hard_negatives=7,
            num_epochs=3,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            global_batch_size=4,
            seq_len=512,
            max_norm=1.0,
            steps=999999,
        ),
        optimizer=OptimizersContainer.Config(
            lr=1e-5,
            weight_decay=0.01,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=50,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            interval=999999,
        ),
    )
