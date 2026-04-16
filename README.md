# ml-template

HUD evaluation environment for ML training tasks. 10 tasks across embedding retrieval, VLM, Flux diffusion, and MoE language models. All run on 1x H100 80GB via [pytorch/torchtitan](https://github.com/pytorch/torchtitan).

## Setup

```bash
git clone git@github.com:hud-evals/ml-template.git
cd ml-template

uv sync
```

## Running Tasks

```bash
# Deploy once
uv run modal deploy modal_runner.py

# Single task
uv run python modal_runner.py --task emb_debug_multi

# Multiple tasks in parallel
uv run python modal_runner.py --tasks emb_debug_multi,moe_debug_balance,flux_debug_timestep

# All tasks, 4 repeats each
uv run python modal_runner.py --all --repeats 4 --model claude-opus-4-6

# Evaluate a checkpoint on MTEB + local evals
uv run python modal_runner.py --eval-checkpoint assets/checkpoints/scifact_base --eval-benchmarks SciFact --eval-local data/val.jsonl
```

## Running Tests

```bash
# Structural tests (no GPU)
uv run pytest tasks/tests/ -v

# GPU tests on Modal
uv run modal deploy modal_runner.py
uv run python modal_runner.py --test
uv run python modal_runner.py --test --test-filter emb
```

## Build & Deploy

```bash
hud deploy .
hud sync tasks <taskset-name>
```

## Architecture

Private [pytorch/torchtitan](https://github.com/pytorch/torchtitan) mirror with HUD evaluation layer on top.

```
env.py                    # Scenarios, grading harness, tool registration
modal_runner.py           # Modal orchestrator (deploy, run tasks, tests, evals)
torchtitan/               # Framework source (upstream fork)
tasks/
├── <slug>/task.py        # One package per task (prompt, graders, scenario args)
├── graders/              # Reusable grader scripts (executed at grade time)
├── mutations/            # Data/eval mutations (copied into Docker image)
├── utils/                # Dataset builders, setup fixtures (copied into Docker image)
└── tests/                # Structural + GPU integration tests
```

`env.py` defines scenarios and the grading harness. Grader scripts are embedded as strings in task args at import time and written to `/tmp/` at grade time -- no direct imports from `tasks/` at runtime.

### What we own vs upstream

| Layer | Paths | Notes |
|-------|-------|-------|
| **HUD evaluation** | `env.py`, `tasks/`, `Dockerfile.hud`, `modal_runner.py` | Safe to modify freely |
| **Experiments** | `torchtitan/experiments/embedding/`, `torchtitan/experiments/vlm/` | Extend via `forward_backward_step` override |
| **Model configs** | `torchtitan/models/flux/diagnostics.py`, `torchtitan/models/flux/config_registry.py` | Flux debug configs + diagnostics |

Everything else under `torchtitan/` is upstream. Pull fixes via `git merge upstream/main`.
