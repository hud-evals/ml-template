"""Evaluate VLM training quality by computing validation loss.

Provides ``evaluate_vlm`` as a library function.  Can also be invoked
as a script::

    python -m torchtitan.experiments.vlm.evaluate \
        --checkpoint_dir checkpoints/vlm --data_path data/cc12m \
        --tokenizer_path tokenizer --steps 20
"""

from __future__ import annotations

import json
import logging
import math
import os

import torch

from torchtitan.experiments.vlm.model.args import VLMTokenNames

logger = logging.getLogger(__name__)


def _ensure_single_gpu_env() -> None:
    """Set distributed env-vars for single-GPU evaluation if not already set."""
    defaults = {
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": "1",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "29501",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)


def evaluate_vlm(
    checkpoint_dir: str,
    tokenizer_path: str,
    *,
    flavor: str = "debugmodel",
    data_path: str | None = None,
    dataset: str = "cc12m-test",
    steps: int = 20,
    batch_size: int = 4,
    seq_len: int = 2048,
    vlm_token_names: "VLMTokenNames | None" = None,
) -> dict:
    """Run validation on a VLM checkpoint and return metrics dict.

    Metrics are also saved to ``<checkpoint_dir>/eval_metrics.json``.
    """
    _ensure_single_gpu_env()

    from torchtitan.components.checkpoint import CheckpointManager
    from torchtitan.components.lr_scheduler import LRSchedulersContainer
    from torchtitan.components.metrics import MetricsProcessor
    from torchtitan.components.optimizer import OptimizersContainer
    from torchtitan.config import ActivationCheckpointConfig, TrainingConfig
    from torchtitan.trainer import Trainer

    from . import model_registry
    from .configs import MultiModalTrainerConfig
    from .datasets.mm_datasets import HuggingFaceMultiModalDataLoader

    dl_kwargs: dict = dict(dataset=dataset, dataset_path=data_path, infinite=True)
    if vlm_token_names is not None:
        dl_kwargs["vlm_token_names"] = vlm_token_names

    config = MultiModalTrainerConfig(
        hf_assets_path=tokenizer_path,
        model_spec=model_registry(flavor),
        dump_folder=checkpoint_dir,
        training=TrainingConfig(
            local_batch_size=batch_size,
            global_batch_size=batch_size,
            seq_len=seq_len,
            steps=steps,
        ),
        dataloader=HuggingFaceMultiModalDataLoader.Config(**dl_kwargs),
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
        for _ in range(steps):
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

    metrics_path = os.path.join(checkpoint_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)

    trainer.close()
    return metrics


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    kwargs: dict = {}
    args = sys.argv[1:]
    i = 0
    _INT_KEYS = {"steps", "batch_size", "seq_len"}
    while i < len(args):
        key = args[i].lstrip("-")
        val = args[i + 1]
        if key in _INT_KEYS:
            kwargs[key] = int(val)
        else:
            kwargs[key] = val
        i += 2

    metrics = evaluate_vlm(**kwargs)
    print(json.dumps(metrics, indent=2))
