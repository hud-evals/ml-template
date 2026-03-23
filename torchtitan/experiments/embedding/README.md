# Contrastive Embedding Training in `torchtitan`

Train contrastive embedding models using InfoNCE loss with last-token (EOS) pooling on a Qwen3 backbone.

## Features
- **InfoNCE contrastive loss** with in-batch negatives and false-negative masking
- **Last-token (EOS) pooling** for fixed-size sentence embeddings
- **Matryoshka representation learning** for multi-resolution embeddings
- **Multi-stage pipeline**: pretrain on synthetic data, fine-tune on labeled data, SLERP merge, evaluate via MTEB
- **HF-format checkpoint export** on training completion for direct use with `transformers` and MTEB

## Architecture

Uses `torchtitan.models.qwen3.Qwen3Model` as the backbone. `EmbeddingTrainer` subclasses `Trainer` and overrides only `forward_backward_step` — the intended extension point — to perform a triple forward pass (query, positive, negative) through the shared backbone, extract hidden states before the output projection, pool via last-token position, and compute InfoNCE loss.

The base `Trainer.train()` loop drives training: `EmbeddingDataLoader` yields data for all epochs and the loop terminates on `DataloaderExhaustedError`. On `close()`, the final model is exported in HF format for evaluation.

## Usage

```bash
torchrun --nproc_per_node 1 -m torchtitan.train \
    --module embedding --config scifact_finetune
```

Named configs are defined in `config_registry.py`: `scifact_pretrain`, `scifact_finetune`, `scifact_matryoshka`. Any field can be overridden on the CLI via tyro (e.g. `--dump_folder ./my_output --dataloader.num_epochs 5`).

## Data Format

Training data is JSONL with one record per line:

```json
{
  "instruction": "Given a scientific claim, retrieve evidence that supports or refutes it",
  "query": "...",
  "positive": "...",
  "negatives": ["...", "...", "..."]
}
```

Training data is JSONL generated from raw HuggingFace datasets (BeIR/SciFact). Model weights and tokenizer are downloaded via `scripts/download_hf_assets.py` to `assets/hf/Qwen3-0.6B/`.

## Pipeline

1. **Pretrain** on synthetic query-passage pairs (`synthetic.jsonl`)
2. **Fine-tune** on labeled retrieval data (`scifact.jsonl`)
3. **Merge** checkpoints via SLERP interpolation (`torchtitan.experiments.embedding.merge`)
4. **Evaluate** on MTEB benchmarks (`evaluate_local`/`evaluate_mteb`)

## Components

| File | Purpose | Torchtitan core reuse |
|------|---------|----------------------|
| `__init__.py` | `model_registry()` returning `ModelSpec` | `ModelSpec`, `qwen3_configs`, `parallelize_qwen3` |
| `embedding_trainer.py` | `EmbeddingTrainer(Trainer)` — overrides `forward_backward_step` and `close` | `Trainer`, `CheckpointManager`, `BaseTokenizer` |
| `datasets.py` | `EmbeddingDataLoader(BaseDataLoader)` with multi-epoch iteration and `state_dict`/`load_state_dict` | `BaseDataLoader` |
| `losses.py` | InfoNCE + Matryoshka loss | `LossFunctionBuilder` signature |
| `configs.py` | `EmbeddingConfig` dataclass | — |
| `config_registry.py` | Named configs for `ConfigManager` | `TrainingConfig`, optimizer/scheduler configs |
| `evaluate.py` | `evaluate_local` and `evaluate_mteb` library functions | — |
| `merge.py` | `slerp_merge` library function | — |
