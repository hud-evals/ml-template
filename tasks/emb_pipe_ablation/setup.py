"""Build 3-stage pipeline checkpoints + pipeline_result.json."""

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("SRC_DIR", "/mcp_server"))
from tasks.utils.setup_fixtures import train, _write_json


def main(ws: str) -> None:
    if (Path(ws) / "pipeline_result.json").exists():
        return

    common = {
        "embedding.eval_data": "data/val.jsonl", "training.local_batch_size": 4,
        "training.global_batch_size": 16, "training.seq_len": 256,
        "embedding.num_hard_negatives": 4, "dataloader.num_hard_negatives": 4, "training.steps": 40,
    }
    stage1, stage2, stage3 = "checkpoints/stage1_pretrain", "checkpoints/stage2_finetune", "checkpoints/stage3_merged"

    ckpt1 = train(ws, "scifact_pretrain", stage1, **{"dataloader.num_epochs": 1, **common})
    ckpt2 = train(ws, "scifact_finetune", stage2, **{
        "embedding.resume_from": ckpt1, "checkpoint.initial_load_path": ckpt1,
        "dataloader.num_epochs": 1, "optimizer.lr": 1e-7, **common,
    })

    import subprocess
    venv = f"{ws}/.venv/bin"
    env = {**os.environ, "HOME": ws, "PATH": f"{venv}:{os.environ.get('PATH', '')}", "PYTHONPATH": ws}
    subprocess.run([f"{venv}/python", "-m", "torchtitan.experiments.embedding.merge",
                    "--checkpoints", ckpt1, ckpt2, "--weights", "0.5", "0.5",
                    "--output_dir", os.path.join(ws, stage3)], cwd=ws, env=env, check=True, capture_output=True, timeout=2400)

    sys.path.insert(0, ws)
    from torchtitan.experiments.embedding.evaluate import evaluate_local
    metrics = evaluate_local(str(Path(ws) / stage3), str(Path(ws) / "data" / "val.jsonl"), max_seq_length=256)
    _write_json(Path(ws) / "pipeline_result.json", {
        "ndcg@10": round(metrics.get("ndcg@10", 0.0), 4),
        "pipeline_stages": [
            {"name": "pretrain", "checkpoint": stage1, "data": "data/synthetic.jsonl"},
            {"name": "finetune", "checkpoint": stage2, "data": "data/scifact.jsonl", "resume_from": stage1, "learning_rate": 1e-7},
            {"name": "merge", "checkpoint": stage3, "method": "SLERP", "weights": [0.5, 0.5]},
        ],
    })


if __name__ == "__main__":
    main(sys.argv[1])
