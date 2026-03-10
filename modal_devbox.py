"""Modal devbox for embedding training experiments.

Setup:
    pip install modal
    modal setup
    modal secret create lib-github-pat LIB_GITHUB_PAT=<pat>
    modal secret create hud-keys HUD_API_KEY=<key>

Usage:
    modal shell modal_devbox.py::dev
    modal run modal_devbox.py::smoke_test
    modal run modal_devbox.py::run_agent
"""

import os

import modal

app = modal.App("ml-training-dev")

vol = modal.Volume.from_name("ml-training-workspace", create_if_missing=True)

lib_pat_secret = modal.Secret.from_name("lib-github-pat", required_keys=["LIB_GITHUB_PAT"])

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "vim", "curl")
    .pip_install(
        "torch>=2.2.0",
        "torchdata>=0.8.0",
        "transformers>=4.40.0,<5",
        "datasets>=2.14,<3.0",
        "mteb>=1.12,<2",
        "hud-python>=0.5.28",
        "pytest",
        "openai",
        "tyro",
        "tensorboard",
        "einops",
        "safetensors",
        "tokenizers>=0.15.0",
    )
    .run_commands(
        "bash -c 'pip install \"hud-sdlc-lib @ git+https://${LIB_GITHUB_PAT}@github.com/hud-evals/hud-sdlc-lib.git@main\"'",
        secrets=[lib_pat_secret],
    )
    .run_commands(
        # Pre-cache base model weights
        "python -c \""
        "from transformers import AutoModel, AutoTokenizer; "
        "AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True); "
        "AutoModel.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True)"
        "\"",
        # Pre-cache embedding model (for baseline/golden benchmarking)
        "python -c \""
        "from transformers import AutoModel, AutoTokenizer; "
        "AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True); "
        "AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True)"
        "\"",
        # Pre-cache MTEB eval datasets (SciFact)
        "python -c \""
        "import mteb; "
        "tasks = mteb.get_tasks(tasks=['SciFact']); "
        "[t.load_data() for t in tasks]"
        "\"",
        # Pre-cache BeIR datasets used by prepare_data
        "python -c \""
        "from datasets import load_dataset; "
        "load_dataset('BeIR/scifact', 'corpus', split='corpus', trust_remote_code=True); "
        "load_dataset('BeIR/scifact', 'queries', split='queries', trust_remote_code=True); "
        "load_dataset('BeIR/scifact-qrels', split='train', trust_remote_code=True); "
        "load_dataset('BeIR/scifact-qrels', split='test', trust_remote_code=True); "
        "load_dataset('sentence-transformers/natural-questions', split='train')"
        "\"",
    )
    .add_local_dir("torchtitan", remote_path="/code/torchtitan")
    .add_local_dir("data", remote_path="/code/data")
    .add_local_dir("grading", remote_path="/code/grading")
    .add_local_dir("tasks", remote_path="/code/tasks")
    .add_local_dir("tests", remote_path="/code/tests")
    .add_local_file("env.py", remote_path="/code/env.py")
    .add_local_file("local_test.py", remote_path="/code/local_test.py")
    .add_local_file("smoke_test.py", remote_path="/code/smoke_test.py")
)


def _sync_workspace():
    """Copy code into the writable workspace volume."""
    import shutil
    import subprocess

    os.makedirs("/workspace", exist_ok=True)

    dest = "/workspace/torchtitan"
    if os.path.exists(dest):
        shutil.rmtree(dest)
    subprocess.run(["cp", "-r", "/code/torchtitan", dest], check=True)
    os.makedirs("/workspace/data", exist_ok=True)

    if os.path.exists("/workspace/tasks"):
        shutil.rmtree("/workspace/tasks")
    subprocess.run(["cp", "-r", "/code/tasks", "/workspace/tasks"], check=True)

    if os.path.exists("/workspace/grading"):
        shutil.rmtree("/workspace/grading")
    subprocess.run(["cp", "-r", "/code/grading", "/workspace/grading"], check=True)

    if os.path.exists("/workspace/data_pkg"):
        shutil.rmtree("/workspace/data_pkg")
    subprocess.run(["cp", "-r", "/code/data", "/workspace/data_pkg"], check=True)

    if os.path.exists("/workspace/tests"):
        shutil.rmtree("/workspace/tests")
    subprocess.run(["cp", "-r", "/code/tests", "/workspace/tests"], check=True)

    for f in ["env.py", "local_test.py", "smoke_test.py"]:
        subprocess.run(["cp", f"/code/{f}", f"/workspace/{f}"], check=True)

    os.chdir("/workspace")


@app.function(
    image=image,
    gpu="H100",
    volumes={"/workspace": vol},
    timeout=7200,
    secrets=[modal.Secret.from_name("hud-keys", required_keys=["HUD_API_KEY"])],
)
def dev():
    """Interactive dev shell with GPU."""
    import subprocess

    _sync_workspace()
    print("=== ML Training Devbox (torchtitan) ===")
    print(f"GPU: {os.popen('nvidia-smi --query-gpu=name --format=csv,noheader').read().strip()}")
    print("Workspace: /workspace")
    print("  torchtitan: /workspace/torchtitan/")
    print("  embedding:  /workspace/torchtitan/experiments/embedding/")
    print("  data:       /workspace/data/")
    print("")
    print("Pipeline stages:")
    print("  1. Generate synthetic data:  python -m data.build_datasets synthetic --corpus_dataset scifact --output data/synthetic.jsonl --max_samples 5000")
    print("  2. Pre-train (stage 1):      python -m torchtitan.experiments.embedding.train --stage pretrain --train_data data/synthetic.jsonl --output_dir checkpoints/stage1 --epochs 1 --batch_size 4")
    print("  3. Download labeled data:    python -m data.build_datasets download --dataset scifact --output data/train.jsonl --max_samples 500")
    print("  4. Fine-tune (stage 2):      python -m torchtitan.experiments.embedding.train --stage finetune --resume_from checkpoints/stage1/epoch_1 --train_data data/train.jsonl --output_dir checkpoints/stage2 --epochs 1 --batch_size 4")
    print("  5. Merge checkpoints:        python -m torchtitan.experiments.embedding.merge --checkpoints checkpoints/stage2/epoch_1 checkpoints/stage1/epoch_1 --output_dir checkpoints/merged")
    print("  6. Evaluate:                 python -m torchtitan.experiments.embedding.evaluate --model checkpoints/merged --tasks SciFact")
    print("")

    subprocess.run(
        ["bash", "-l"],
        env={**os.environ, "PYTHONPATH": "/workspace:/code"},
    )

    vol.commit()


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
)
def smoke_test():
    """Run the full 3-stage pipeline with small data as a smoke test."""
    import subprocess

    os.makedirs("/workspace", exist_ok=True)
    _sync_workspace()

    env = {**os.environ, "PYTHONPATH": "/workspace:/code"}
    run = lambda cmd: subprocess.run(cmd, cwd="/workspace", env=env, check=True)

    print("=== Stage 1: Generate synthetic data ===")
    run([
        "python", "-m", "data.build_datasets", "synthetic",
        "--corpus_dataset", "scifact",
        "--output", "data/synthetic.jsonl",
        "--max_samples", "500",
    ])

    print("\n=== Stage 1: Pre-train ===")
    run([
        "python", "-m", "torchtitan.experiments.embedding.train",
        "--stage", "pretrain",
        "--train_data", "data/synthetic.jsonl",
        "--output_dir", "checkpoints/stage1",
        "--model", "Qwen/Qwen3-0.6B",
        "--epochs", "1",
        "--batch_size", "4",
        "--max_seq_length", "256",
    ])

    print("\n=== Stage 2: Download labeled data ===")
    run([
        "python", "-m", "data.build_datasets", "download",
        "--dataset", "scifact",
        "--output", "data/train.jsonl",
        "--max_samples", "200",
    ])

    print("\n=== Stage 2: Fine-tune from stage 1 ===")
    run([
        "python", "-m", "torchtitan.experiments.embedding.train",
        "--stage", "finetune",
        "--resume_from", "checkpoints/stage1/epoch_1",
        "--train_data", "data/train.jsonl",
        "--output_dir", "checkpoints/stage2",
        "--model", "Qwen/Qwen3-0.6B",
        "--epochs", "1",
        "--batch_size", "4",
        "--max_seq_length", "256",
    ])

    print("\n=== Stage 3: SLERP merge ===")
    run([
        "python", "-m", "torchtitan.experiments.embedding.merge",
        "--checkpoints", "checkpoints/stage1/epoch_1", "checkpoints/stage2/epoch_1",
        "--output_dir", "checkpoints/merged",
    ])

    print("\n=== Evaluate merged model ===")
    run([
        "python", "-m", "torchtitan.experiments.embedding.evaluate",
        "--model", "checkpoints/merged",
        "--tasks", "SciFact",
        "--max_seq_length", "256",
    ])


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
)
def vlm_smoke_test():
    """Run VLM training + evaluation as a smoke test."""
    import subprocess

    os.makedirs("/workspace", exist_ok=True)
    _sync_workspace()

    env = {**os.environ, "PYTHONPATH": "/workspace:/code"}
    run = lambda cmd: subprocess.run(cmd, cwd="/workspace", env=env, check=True)

    print("=== VLM: Train (50 steps) ===")
    run([
        "python", "-m", "torchtitan.experiments.vlm.train",
        "--dataset", "cc12m-test",
        "--data_path", "/workspace/tests/assets/cc12m_test",
        "--tokenizer_path", "/workspace/tests/assets/tokenizer",
        "--output_dir", "/workspace/checkpoints/vlm",
        "--steps", "50",
        "--batch_size", "4",
    ])

    print("\n=== VLM: Evaluate ===")
    run([
        "python", "-m", "torchtitan.experiments.vlm.evaluate",
        "--checkpoint_dir", "/workspace/checkpoints/vlm",
        "--data_path", "/workspace/tests/assets/cc12m_test",
        "--dataset", "cc12m-test",
        "--tokenizer_path", "/workspace/tests/assets/tokenizer",
        "--steps", "10",
        "--batch_size", "4",
    ])

    import json
    metrics_path = "/workspace/checkpoints/vlm/eval_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        print(f"\nVLM eval results: {json.dumps(metrics, indent=2)}")
        val_loss = metrics.get("val_loss", 999)
        assert val_loss < 5.0, f"VLM val_loss {val_loss} too high (expected < 5.0)"
        print(f"PASS: val_loss = {val_loss}")
    else:
        raise FileNotFoundError("eval_metrics.json not produced")


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
)
def grading_smoke_test():
    """Run the grading infrastructure smoke test on Modal."""
    import subprocess

    os.makedirs("/workspace", exist_ok=True)
    _sync_workspace()

    env = {**os.environ, "PYTHONPATH": "/workspace:/code"}
    result = subprocess.run(
        ["python", "-m", "pytest", "/workspace/smoke_test.py", "-v", "--tb=short"],
        cwd="/workspace",
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError("Grading smoke test failed")
    print("All grading smoke tests passed")


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
)
def benchmark():
    """Evaluate baseline (raw LM) and golden (embedding model) on SciFact."""
    import subprocess

    os.makedirs("/workspace", exist_ok=True)
    _sync_workspace()

    env = {**os.environ, "PYTHONPATH": "/workspace:/code"}

    for label, model in [
        ("Baseline (Qwen/Qwen3-0.6B, untrained LM)", "Qwen/Qwen3-0.6B"),
        ("Golden (Qwen/Qwen3-Embedding-0.6B)", "Qwen/Qwen3-Embedding-0.6B"),
    ]:
        print(f"\n=== {label} ===")
        result = subprocess.run(
            [
                "python", "-c",
                "import sys; sys.path.insert(0, '.'); "
                f"from torchtitan.experiments.embedding.evaluate import evaluate_mteb; "
                f"import json; m = evaluate_mteb('{model}', ['SciFact']); "
                f"print(json.dumps(m, indent=2))",
            ],
            cwd="/workspace", env=env, capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print("STDERR:", result.stderr[-1000:])


@app.function(
    image=image,
    gpu="H100",
    timeout=86400,
    secrets=[modal.Secret.from_name("hud-keys", required_keys=["HUD_API_KEY"])],
)
def run_agent(task: str = "finetune_embedding", model: str = "claude-opus-4-6", max_steps: int = 500):
    """Run an agent against a specific task."""
    import subprocess

    os.makedirs("/workspace", exist_ok=True)
    _sync_workspace()

    print(f"=== Running agent: task={task}, model={model}, max_steps={max_steps} ===")
    subprocess.run(
        [
            "python", "local_test.py",
            "--task", task,
            "--model", model,
            "--max-steps", str(max_steps),
        ],
        cwd="/workspace",
        env={**os.environ, "PYTHONPATH": "/workspace:/code", "MCP_TESTING_MODE": "1"},
    )


@app.local_entrypoint()
def trigger(
    task: str = "finetune_embedding",
    model: str = "claude-opus-4-6",
    max_steps: int = 500,
):
    """CLI entrypoint: run or spawn an agent against a task.

    Usage:
        modal run modal_devbox.py --task vlm_finetune
        modal run modal_devbox.py --task debug_embedding_loss --model grok-4-1-fast
        modal run modal_devbox.py --task finetune_embedding --max-steps 200
    """
    print(f"Running: task={task}, model={model}, max_steps={max_steps}")
    run_agent.remote(task=task, model=model, max_steps=max_steps)
