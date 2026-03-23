# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import ActivationCheckpointConfig, TrainingConfig

from . import model_registry

from .configs import MultiModalTrainerConfig
from .datasets.mm_datasets import HuggingFaceMultiModalDataLoader
from .model.args import VLMTokenNames

_DEBUG_TOKEN_NAMES = VLMTokenNames(
    img="<|image|>",
    boi="<|begin_of_image|>",
    eoi="<|end_of_image|>",
    pad="<|pad|>",
)


def vlm_qwen3_0_6B() -> MultiModalTrainerConfig:
    """Qwen3-0.6B VLM config for real training runs."""
    return MultiModalTrainerConfig(
        hf_assets_path="./assets/hf/Qwen3-0.6B",
        model_spec=model_registry("qwen3_0.6B"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=10,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=2048,
            steps=100,
        ),
        dataloader=HuggingFaceMultiModalDataLoader.Config(dataset="cc12m-test", dataset_path="data/cc12m"),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpoint=CheckpointManager.Config(
            enable=True,
            interval=100,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
    )


def vlm_debugmodel() -> MultiModalTrainerConfig:
    return MultiModalTrainerConfig(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("debugmodel"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        dataloader=HuggingFaceMultiModalDataLoader.Config(
            dataset="cc12m-test",
            vlm_token_names=_DEBUG_TOKEN_NAMES,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
    )
