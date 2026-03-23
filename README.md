# ml-template

HUD evaluation environment for ML training tasks. 10 tasks across embedding retrieval, VLM, Flux diffusion, and MoE language models. All run on 1x H100 80GB via [pytorch/torchtitan](https://github.com/pytorch/torchtitan).

## Setup

```bash
git clone git@github.com:hud-evals/ml-template.git
cd ml-template
cp .env.example .env
# Fill in LIB_GITHUB_PAT, HUD_API_KEY

uv sync
```

## Running Tasks

### Locally (requires GPU)

```bash
# List tasks
PYTHONPATH=. uv run python local_test.py --list

# Run a task with a specific model
PYTHONPATH=. uv run python local_test.py --task emb_debug_multi
PYTHONPATH=. uv run python local_test.py --task emb_efficient --model claude-opus-4-6 --max-steps 200
```

### On Modal (no local GPU needed)

```bash
# Deploy once
uv run modal deploy modal_devbox.py

# Single task
uv run python modal_devbox.py --task emb_debug_multi

# Multiple tasks in parallel
uv run python modal_devbox.py --tasks emb_debug_multi,moe_debug_balance,flux_debug_timestep

# All tasks, 4 repeats each
uv run python modal_devbox.py --all --repeats 4 --model claude-opus-4-6

# Evaluate a checkpoint on MTEB + local evals
uv run python modal_devbox.py --eval-checkpoint assets/checkpoints/scifact_base --eval-benchmarks SciFact --eval-local data/val.jsonl
```

## Running Tests

```bash
# Structural tests (no GPU)
uv run pytest tasks/tests/ -v

# GPU integration tests locally
uv run pytest tasks/tests/tasks/ -v

# GPU tests on Modal
uv run modal deploy modal_devbox.py
uv run python modal_devbox.py --test
uv run python modal_devbox.py --test --test-filter emb
```

## Build & Deploy

```bash
uv run build          # Build Docker image
uv run deploy         # Deploy to HUD platform
uv run sync-tasks     # Sync task definitions
```

## Architecture

Private [pytorch/torchtitan](https://github.com/pytorch/torchtitan) mirror with HUD evaluation layer on top.

```
tasks/
├── <slug>/task.py        # One package per task (prompt, graders, scenario args)
├── graders/              # Reusable grader scripts (executed at grade time)
├── mutations/            # Data/eval mutations + shared CLI entry point
├── utils/                # Dataset builders, setup fixtures
└── tests/                # Structural + GPU integration tests
```

`env.py` defines scenarios and the grading harness. Grader scripts run as subprocesses -- no direct imports from `tasks/` at runtime.

### What we own vs upstream

| Layer | Paths | Notes |
|-------|-------|-------|
| **HUD evaluation** | `env.py`, `tasks/`, `Dockerfile.hud`, `local_test.py`, `modal_devbox.py` | Safe to modify freely |
| **Experiments** | `torchtitan/experiments/embedding/`, `torchtitan/experiments/vlm/` | Extend via `forward_backward_step` override |
| **Model configs** | `torchtitan/models/flux/diagnostics.py`, `torchtitan/models/flux/config_registry.py` | Flux debug configs + diagnostics |

Everything else under `torchtitan/` is upstream. Pull fixes via `git merge upstream/main`.
