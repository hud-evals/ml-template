"""CLI entry point for embedding training.

Standalone usage (for agents):
    python -m torchtitan.experiments.embedding.train \
        --stage pretrain --train_data data/synthetic.jsonl \
        --output_dir checkpoints/stage1 --epochs 1 --batch_size 4

Torchtitan CLI (via torchrun):
    torchrun --nproc_per_node 1 -m torchtitan.train \
        --module embedding --config scifact_finetune
"""

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune embedding model")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--eval_data", default=None)
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--stage", choices=["pretrain", "finetune"], default="finetune")
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--num_hard_negatives", type=int, default=7)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--false_neg_threshold", type=float, default=0.1)
    parser.add_argument("--matryoshka_dims", type=int, nargs="*", default=None)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=None)
    args = parser.parse_args()

    # Set up single-GPU distributed environment
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    from huggingface_hub import snapshot_download

    from torchtitan.components.checkpoint import CheckpointManager
    from torchtitan.components.lr_scheduler import LRSchedulersContainer
    from torchtitan.components.optimizer import OptimizersContainer
    from torchtitan.config.configs import TrainingConfig

    from . import model_registry
    from .configs import EmbeddingConfig
    from .datasets import EmbeddingDataLoader
    from .embedding_trainer import EmbeddingTrainer

    # Resolve model path (download if not cached)
    model_source = args.resume_from or args.model
    if os.path.isdir(model_source):
        model_path = model_source
    else:
        model_path = snapshot_download(model_source)

    # Derive torchtitan flavor from HF model name (e.g. "Qwen/Qwen3-1.7B" -> "1.7B")
    _FLAVOR_MAP = {"Qwen/Qwen3-0.6B": "0.6B", "Qwen/Qwen3-1.7B": "1.7B", "Qwen/Qwen3-4B": "4B", "Qwen/Qwen3-8B": "8B"}
    flavor = _FLAVOR_MAP.get(args.model, "0.6B")

    config = EmbeddingTrainer.Config(
        model_spec=model_registry(flavor),
        hf_assets_path=model_path,
        dump_folder=args.output_dir,
        embedding=EmbeddingConfig(
            stage=args.stage,
            model_name=args.model,
            resume_from=args.resume_from,
            temperature=args.temperature,
            num_hard_negatives=args.num_hard_negatives,
            false_negative_threshold=args.false_neg_threshold,
            matryoshka_dims=args.matryoshka_dims,
            num_epochs=args.epochs,
            save_steps=args.save_steps,
            train_data=args.train_data,
            eval_data=args.eval_data,
        ),
        dataloader=EmbeddingDataLoader.Config(
            train_path=args.train_data,
            model_name=args.model,
            num_hard_negatives=args.num_hard_negatives,
        ),
        training=TrainingConfig(
            local_batch_size=args.batch_size,
            global_batch_size=args.batch_size * args.gradient_accumulation_steps,
            seq_len=args.max_seq_length,
            max_norm=args.max_grad_norm,
            steps=999999,
        ),
        optimizer=OptimizersContainer.Config(
            lr=args.lr,
            weight_decay=args.weight_decay,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=int(args.warmup_ratio * 100),
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            initial_load_path=args.resume_from if args.resume_from else None,
            interval=999999,
        ),
    )

    trainer = EmbeddingTrainer(config=config)
    trainer.train()
    trainer.close()


if __name__ == "__main__":
    main()
