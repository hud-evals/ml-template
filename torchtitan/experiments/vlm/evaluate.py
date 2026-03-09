"""Evaluate VLM training quality by computing validation loss.

Usage:
    python -m torchtitan.experiments.vlm.evaluate \
        --checkpoint_dir checkpoints/vlm --data_path data/cc12m \
        --tokenizer_path tokenizer --steps 20
"""

import argparse
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM checkpoint")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--flavor", default="debugmodel",
                        help="Model flavor: debugmodel, qwen3_0.6B, qwen3_1.7B, etc.")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--dataset", default="cc12m-test")
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=2048)
    args = parser.parse_args()

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")

    import math

    import torch
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
        model_spec=model_registry(args.flavor),
        dump_folder=args.checkpoint_dir,
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
        optimizer=OptimizersContainer.Config(lr=0.0),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=0),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpoint=CheckpointManager.Config(
            enable=True,
            interval=999999,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
    )

    trainer = Trainer(config=config)
    trainer.checkpointer.load(step=config.checkpoint.load_step)

    model = trainer.model_parts[0]
    model.eval()

    total_loss = 0.0
    count = 0
    data_iter = trainer.batch_generator(trainer.dataloader)

    with torch.no_grad():
        for _ in range(args.steps):
            try:
                input_dict, labels = next(data_iter)
                for k, v in input_dict.items():
                    if isinstance(v, torch.Tensor):
                        input_dict[k] = v.to("cuda")
                labels = labels.to("cuda")

                inputs, labels_proc, extra_inputs, extra_kwargs = (
                    trainer.post_dataloading_process(input_dict, labels)
                )
                pred = model(inputs, **extra_inputs, **extra_kwargs)
                loss = torch.nn.functional.cross_entropy(
                    pred.flatten(0, -2), labels_proc.flatten(0, -1)
                )
                total_loss += loss.item()
                count += 1
            except StopIteration:
                break

    avg_loss = total_loss / max(count, 1)
    perplexity = math.exp(min(avg_loss, 20.0))

    metrics = {
        "val_loss": round(avg_loss, 4),
        "perplexity": round(perplexity, 2),
        "eval_steps": count,
    }
    logger.info("Evaluation results: %s", json.dumps(metrics, indent=2))

    metrics_path = os.path.join(args.checkpoint_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)

    trainer.close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
