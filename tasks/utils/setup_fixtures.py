"""Download assets and build fixtures for task setup.

Usage:
    python setup_fixtures.py <workspace> --models Qwen/Qwen3-0.6B --datasets scifact nq
    python setup_fixtures.py <workspace> --checkpoints scifact_base
    python setup_fixtures.py <workspace> --data-files pixparse/cc12m-wds cc12m-train-0000.tar cc12m-train-0001.tar cc12m-train-0002.tar
    python setup_fixtures.py <workspace> --train scifact_finetune checkpoints/out --train-overrides training.steps=60
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

SRC = os.environ.get("SRC_DIR", "/mcp_server")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def download_model(ws: str, repo_id: str) -> Path:
    """Download an HF model (tokenizer + weights + config) into ws/assets/hf/."""
    name = repo_id.split("/")[-1]
    dest = Path(ws) / "assets" / "hf" / name
    if dest.is_dir():
        return dest
    local = Path(SRC) / "assets" / "hf" / name
    if local.is_dir():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(local), str(dest))
        return dest
    sys.path.insert(0, SRC)
    from scripts.download_hf_assets import download_hf_assets
    dest.parent.mkdir(parents=True, exist_ok=True)
    # download_hf_assets creates a subdirectory named after the model, so pass the parent
    download_hf_assets(repo_id, str(dest.parent), asset_types=["tokenizer", "safetensors", "config"])
    return dest


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

_DATASET_LOADERS = {
    "scifact": ("load_scifact", 500, 7, 50, "val"),
    "nq": ("load_nq", 500, 7, 50, "nq_val"),
}


def build_dataset(ws: str, name: str) -> None:
    """Build a training dataset from HuggingFace sources."""
    data_dir = Path(ws) / "data"
    if (data_dir / f"{name}.jsonl").exists():
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, SRC)

    if name == "synthetic":
        from tasks.utils.build_datasets import generate_synthetic
        pairs = generate_synthetic(corpus_dataset="scifact", max_samples=500)
        with open(data_dir / "synthetic.jsonl", "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        return

    if name not in _DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {sorted(_DATASET_LOADERS)} + synthetic")

    loader_name, max_samples, num_negatives, val_samples, val_prefix = _DATASET_LOADERS[name]
    import tasks.utils.build_datasets as bd
    pairs = getattr(bd, loader_name)(max_samples=max_samples, num_negatives=num_negatives)
    random.seed(42)
    random.shuffle(pairs)

    with open(data_dir / f"{name}.jsonl", "w") as f:
        for p in pairs[val_samples:]:
            f.write(json.dumps(p) + "\n")
    with open(data_dir / f"{val_prefix}.jsonl", "w") as f:
        for p in pairs[:val_samples]:
            f.write(json.dumps(p) + "\n")


# ---------------------------------------------------------------------------
# Data files (download specific files from HF dataset repos)
# ---------------------------------------------------------------------------


def download_data_files(ws: str, repo: str, files: list[str], dest_name: str | None = None) -> Path:
    """Download specific files from an HF dataset repo into ws/data/<name>/."""
    name = dest_name or repo.split("/")[-1].lower().replace("-wds", "")
    dest = Path(ws) / "data" / name
    if dest.is_dir() and len(list(dest.iterdir())) >= len(files):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    for f in files:
        hf_hub_download(repo, f, local_dir=str(dest), repo_type="dataset")
    return dest


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def build_checkpoint(ws: str, name: str) -> Path:
    """Build a pre-trained checkpoint from downloaded models."""
    dest = Path(ws) / "assets" / "checkpoints" / name
    if dest.is_dir():
        return dest

    if name == "scifact_base":
        qwen = download_model(ws, "Qwen/Qwen3-0.6B")
        download_model(ws, "Qwen/Qwen3-Embedding-0.6B")
        emb_dir = download_model(ws, "LinerAI/Qwen3-Embedding-0.6B-academic")

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(emb_dir), str(dest))
        shutil.copy2(str(qwen / "config.json"), str(dest / "config.json"))
        for f in qwen.glob("tokenizer*.json"):
            shutil.copy2(str(f), str(dest))
        for fname in ("vocab.json", "merges.txt"):
            src = qwen / fname
            if src.exists():
                shutil.copy2(str(src), str(dest))

        import torch
        from safetensors.torch import load_file, save_file
        sd = load_file(str(dest / "model.safetensors"))
        nsd = {("model." + k if not k.startswith("model.") else k): v for k, v in sd.items()}
        old = nsd["model.embed_tokens.weight"]
        new = torch.zeros(151936, old.shape[1], dtype=old.dtype)
        new[:old.shape[0]] = old
        nsd["model.embed_tokens.weight"] = new
        save_file(nsd, str(dest / "model.safetensors"))

        shutil.rmtree(str(Path(ws) / "assets" / "hf" / "Qwen3-Embedding-0.6B"), ignore_errors=True)
        shutil.rmtree(str(emb_dir), ignore_errors=True)
    else:
        raise ValueError(f"Unknown checkpoint: {name}")

    return dest


# ---------------------------------------------------------------------------
# Training (runs torchrun, returns checkpoint path)
# ---------------------------------------------------------------------------


def train(ws: str, config: str, dump_folder: str, **overrides: object) -> str:
    """Run embedding training via torchrun. Returns path to latest checkpoint."""
    import glob as _glob
    import subprocess

    venv = f"{ws}/.venv/bin"
    cmd = [f"{venv}/torchrun", "--nproc_per_node", "1", "-m", "torchtitan.train",
           "--module", "embedding", "--config", config, "--dump_folder", dump_folder]
    for k, v in overrides.items():
        if isinstance(v, bool):
            cmd.append(f"--{k.replace('_', '-')}" if v else f"--no-{k.replace('_', '-')}")
        else:
            cmd.extend([f"--{k}", str(v)])

    env = {**os.environ, "HOME": ws, "PATH": f"{venv}:{os.environ.get('PATH', '')}", "PYTHONPATH": ws}
    result = subprocess.run(cmd, cwd=ws, env=env, text=True, capture_output=True, timeout=2400, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed:\nstderr:\n{result.stderr[-4000:]}")

    ckpt_base = os.path.join(ws, dump_folder, "checkpoint")
    step_dirs = sorted(_glob.glob(os.path.join(ckpt_base, "step-*")),
                       key=lambda d: int(os.path.basename(d).split("-")[1]))
    if not step_dirs:
        raise FileNotFoundError(f"No checkpoints in {ckpt_base}")
    ckpt_dir = step_dirs[-1]

    hf_assets = os.path.join(ws, "assets", "hf", "Qwen3-0.6B")
    if os.path.isdir(hf_assets):
        for fname in os.listdir(hf_assets):
            src, dst = os.path.join(hf_assets, fname), os.path.join(ckpt_dir, fname)
            if not os.path.exists(dst) and os.path.isfile(src):
                shutil.copy2(src, dst)
    return ckpt_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workspace")
    parser.add_argument("--models", nargs="+", default=[])
    parser.add_argument("--datasets", nargs="+", default=[])
    parser.add_argument("--data-files", nargs="+", default=[], metavar="REPO_OR_FILE")
    parser.add_argument("--checkpoints", nargs="+", default=[])
    args = parser.parse_args()

    for model in args.models:
        download_model(args.workspace, model)
    for dataset in args.datasets:
        build_dataset(args.workspace, dataset)
    if args.data_files:
        download_data_files(args.workspace, args.data_files[0], args.data_files[1:])
    for ckpt in args.checkpoints:
        build_checkpoint(args.workspace, ckpt)


if __name__ == "__main__":
    main()
