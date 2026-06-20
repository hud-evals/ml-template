# ml-template

HUD evaluation environment for ML training tasks. 10 tasks across embedding retrieval, VLM, Flux diffusion, and MoE language models. All run on 1x H100 80GB via [pytorch/torchtitan](https://github.com/pytorch/torchtitan).

> This environment can be driven three ways: locally, directly on Modal, or
> through the **HUD platform** onto Modal (see [Running through the HUD platform](#running-through-the-hud-platform-modal-runtime)).

## Setup

```bash
git clone git@github.com:hud-evals/ml-template.git
cd ml-template

uv sync
```

## Running Tasks

These commands launch Modal directly from your local machine. The Modal functions
in `modal_runner.py` request `gpu="H100"`.

Modal containers do not inherit local shell exports, and this repo does not
store your HUD API key. Before deploying or running directly on Modal,
authenticate Modal and create a Modal secret with your HUD API key:

```bash
uv run modal setup
uv run modal secret create hud-keys HUD_API_KEY=<your-hud-api-key>
```

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

## Running through the HUD platform (Modal runtime)

The same env can be driven by the **HUD platform** instead of being launched by
hand, so a run is queued and logged on HUD while executing on Modal.

Use `platform_runner.py` for this path. It deploys the environment for hosted
Modal execution and submits runs with the GPU settings this template needs.

Deploy the Modal-backed HUD image and sync tasks:

```bash
uv run python platform_runner.py deploy
uv run hud sync tasks <taskset-name> taskset.py
```

Submit tasks through HUD hosted Modal:

```bash
uv run python platform_runner.py run --task emb_debug_multi --model claude-opus-4-6
uv run python platform_runner.py run --all --group 4 --max-concurrent 4
```

Set `HUD_MODAL_GPU_TYPE` or pass `--gpu-type` if a run needs a different
Modal GPU type. The default is `H100`.

## Architecture

[pytorch/torchtitan](https://github.com/pytorch/torchtitan) fork with HUD evaluation layer on top.

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
