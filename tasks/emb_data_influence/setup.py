"""Build checkpoint_61 + adversarial failing queries."""

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("SRC_DIR", "/mcp_server"))
from setup.setup_fixtures import train, _write_json


def main(ws: str) -> None:
    if (Path(ws) / "failing_queries.json").exists():
        return

    train_path = Path(ws) / "data" / "scifact.jsonl"
    val_path = Path(ws) / "data" / "val.jsonl"

    random.seed(42)
    val_pairs = [json.loads(line) for line in val_path.read_text().splitlines()]
    train_pairs = [json.loads(line) for line in train_path.read_text().splitlines()]
    targets = random.sample(val_pairs, min(10, len(val_pairs)))
    all_positives = [p["positive"] for p in train_pairs]

    with train_path.open("a") as f:
        for pair in targets:
            wrong_pool = [p for p in all_positives if p != pair["positive"]]
            for wrong in random.sample(wrong_pool, min(5, len(wrong_pool))):
                f.write(json.dumps({"query": pair["query"], "positive": wrong,
                                    "instruction": pair.get("instruction", ""),
                                    "negatives": pair.get("negatives", [])}) + "\n")

    queries = [p["query"] for p in targets]
    query_set = set(queries)
    with open(Path(ws) / "data" / "failing_queries_val.jsonl", "w") as f:
        for line in val_path.read_text().splitlines():
            if json.loads(line).get("query") in query_set:
                f.write(line + "\n")
    _write_json(Path(ws) / "failing_queries.json", {
        "queries": queries, "description": "These queries have near-zero nDCG in the staged model.",
    })

    train(ws, "scifact_finetune", "checkpoints/checkpoint_61", **{
        "embedding.train_data": "data/scifact.jsonl", "dataloader.train_path": "data/scifact.jsonl",
        "embedding.eval_data": "data/val.jsonl", "dataloader.num_epochs": 2,
        "training.local_batch_size": 4, "training.global_batch_size": 16, "training.seq_len": 256,
        "embedding.num_hard_negatives": 4, "dataloader.num_hard_negatives": 4, "training.steps": 60,
    })


if __name__ == "__main__":
    main(sys.argv[1])
