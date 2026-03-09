from env import WORKSPACE as W, check_weights, pretrain_embedding

CHECKS = [
    {"name": "has_data",          "command": f"ls {W}/data/*.jsonl 2>/dev/null | head -1"},
    {"name": "pretrain_metadata", "command": f"python /tmp/check_metadata.py pretrain {W}"},
    {"name": "pretrain_weights",  "command": check_weights("pretrain")},
]

task = pretrain_embedding.task(
    prompt=(
        "Pre-train Qwen3-0.6B on synthetic data as the first stage of an embedding pipeline.\n"
        "Evaluate with data/val.jsonl. You choose the hyperparameters."
    ),
    graders=[
        *[{**c, "weight": w} for c, w in zip(CHECKS, [0.10, 0.15, 0.15])],
        {"name": "mteb_eval",     "weight": 0.20, "command": f"python /tmp/grader_eval.py {W}", "timeout": 1200},
        {"name": "ndcg@10>=0.05", "weight": 0.20, "command": f"python /tmp/check_ndcg.py 0.05 {W}"},
        {"name": "ndcg@10>=0.15", "weight": 0.20, "command": f"python /tmp/check_ndcg.py 0.15 {W}"},
    ],
)
task.slug = "pretrain_embedding"
