"""Standalone VLM training CLI.

Usage:
    python -m torchtitan.experiments.vlm.train \
        --dataset cc12m-test --data_path data/cc12m \
        --tokenizer_path tokenizer --output_dir checkpoints/vlm \
        --steps 100 --batch_size 4
"""

import argparse
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune VLM")
    parser.add_argument("--dataset", default="cc12m-test")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output_dir", default="checkpoints/vlm")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--log_freq", type=int, default=1)
    args = parser.parse_args()

    # Set up single-GPU distributed environment
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    from torchtitan.components.checkpoint import CheckpointManager
    from torchtitan.components.lr_scheduler import LRSchedulersContainer
    from torchtitan.components.metrics import MetricsProcessor
    from torchtitan.components.optimizer import OptimizersContainer
    from torchtitan.config import ActivationCheckpointConfig, TrainingConfig
    from torchtitan.trainer import Trainer

    from . import model_registry
    from .configs import MultiModalTrainerConfig
    from .datasets.mm_datasets import HuggingFaceMultiModalDataLoader

    config = MultiModalTrainerConfig(
        hf_assets_path=args.tokenizer_path,
        model_spec=model_registry("debugmodel"),
        dump_folder=args.output_dir,
        training=TrainingConfig(
            local_batch_size=args.batch_size,
            global_batch_size=args.batch_size,
            seq_len=args.seq_len,
            steps=args.steps,
        ),
        dataloader=HuggingFaceMultiModalDataLoader.Config(
            dataset=args.dataset,
            dataset_path=args.data_path,
            infinite=True,
        ),
        optimizer=OptimizersContainer.Config(lr=args.lr),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=args.warmup_steps,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        metrics=MetricsProcessor.Config(log_freq=args.log_freq),
        checkpoint=CheckpointManager.Config(
            enable=True,
            interval=args.steps,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
    )

    trainer = Trainer(config=config)
    trainer.train()

    # Save training metadata
    if int(os.environ.get("RANK", "0")) == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        metadata = {
            "task": "vlm_finetune",
            "model": "debugmodel",
            "dataset": args.dataset,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seq_len": args.seq_len,
        }
        with open(os.path.join(args.output_dir, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        logging.info("Saved training metadata to %s", args.output_dir)

    trainer.close()


if __name__ == "__main__":
    main()
