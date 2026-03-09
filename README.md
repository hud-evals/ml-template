# ml-template

HUD evaluation environment for ML training tasks. Built on a [pytorch/torchtitan](https://github.com/pytorch/torchtitan) mirror with an embedding training experiment and adversarial scenarios.

## Architecture

The repo is a private torchtitan mirror (`upstream` remote) with HUD environment files at root. The agent works with the embedding experiment code in `torchtitan/experiments/embedding/`, training contrastive embedding models via InfoNCE loss on Qwen3-0.6B.

The embedding experiment uses HuggingFace transformers directly (AutoModel, AdamW, save_pretrained) rather than torchtitan's Trainer/Qwen3Model. It lives under `torchtitan/experiments/` for organizational purposes but has its own standalone training loop.

**8 evaluation tasks across 3 types:**
- **Optimization** (4 tasks): pretrain, finetune, merge, full multi-stage pipeline
- **Code debugging** (2 tasks): find and fix bugs in loss function or pooling strategy
- **Data auditing** (2 tasks): detect and clean label noise or train/test leakage

## Setup

```bash
git clone git@github.com:hud-evals/ml-template.git
cd ml-template
cp .env.example .env
# Fill in LIB_GITHUB_PAT, HUD_API_KEY

uv sync
```

## Local GPU Testing

All testing runs locally on a GPU machine. No Docker or cloud services needed.

### 1. Smoke test (verify training pipeline)

```bash
PYTHONPATH=. uv run python -m torchtitan.experiments.embedding.prepare_data download \
    --dataset scifact --output /tmp/test.jsonl --max_samples 100

PYTHONPATH=. uv run python -m torchtitan.experiments.embedding.train \
    --stage finetune --train_data /tmp/test.jsonl \
    --output_dir /tmp/ckpt --epochs 1 --batch_size 2 --max_seq_length 128

PYTHONPATH=. uv run python -m torchtitan.experiments.embedding.evaluate \
    --model /tmp/ckpt/epoch_1 --tasks SciFact
```

### 2. Run agent against a task

```bash
# List available tasks
PYTHONPATH=. uv run python local_test.py --list

# Run a specific task
PYTHONPATH=. uv run python local_test.py --task finetune_embedding
PYTHONPATH=. uv run python local_test.py --task debug_embedding_loss --model grok-4-1-fast
PYTHONPATH=. uv run python local_test.py --task multistage_retrieval --max-steps 200
```

### 3. Docker-based validation (grading verification)

```bash
uv run dev          # Start Docker container with scenario server
uv run validate     # Run baseline-fail + golden-replay validation
```

## Build & Deploy

```bash
uv run build          # Build Docker image
uv run deploy         # Deploy to HUD platform
uv run sync-tasks     # Sync task definitions to platform
```

## Pulling Upstream Updates

```bash
git fetch upstream
git merge upstream/main
# Resolve any conflicts in pyproject.toml, .gitignore
```

## File Reference

| Path | Purpose |
|------|---------|
| `torchtitan/experiments/embedding/` | Embedding training experiment (our code) |
| `env.py` | HUD scenario definitions (6 scenarios) |
| `grading/` | Grading scripts + data mutations (hidden from agent) |
| `tasks/` | Task prompts and scoring configs (8 tasks) |
| `local_test.py` | Run agent against any task locally |
| `Dockerfile.hud` | Docker build for HUD platform |
| `modal_devbox.py` | Modal GPU dev/test (optional) |
| `sdlc_scripts.py` | CLI wrappers for build/deploy/sync |
