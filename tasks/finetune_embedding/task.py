from env import WORKSPACE as W, check_weights, finetune_embedding

CHECKS = [
    {"name": "finetune_metadata", "command": f"python /tmp/check_metadata.py finetune {W}"},
    {"name": "finetune_weights",  "command": check_weights("finetune")},
]

task = finetune_embedding.task(
    prompt=(
        "Fine-tune Qwen3-0.6B into an embedding model that performs well on SciFact retrieval.\n"
        "You choose the hyperparameters and data strategy."
    ),
    graders=[
        {"name": "has_data",          "weight": 0.10, "command": f"ls {W}/data/*.jsonl 2>/dev/null | head -1"},
        *[{**c, "weight": 0.10} for c in CHECKS],
        {"name": "mteb_eval",         "weight": 0.10, "command": f"python /tmp/grader_eval.py {W}", "timeout": 1200},
        {"name": "ndcg@10>=0.10",     "weight": 0.15, "command": "python /tmp/check_ndcg.py 0.10"},
        {"name": "ndcg@10>=0.30",     "weight": 0.15, "command": "python /tmp/check_ndcg.py 0.30"},
        {"name": "ndcg@10>=0.50",     "weight": 0.15, "command": "python /tmp/check_ndcg.py 0.50"},
        {"name": "ndcg@10>=0.65",     "weight": 0.15, "command": "python /tmp/check_ndcg.py 0.65"},
    ],
)
task.slug = "finetune_embedding"
