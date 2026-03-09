from env import WORKSPACE as W, check_merge_weights, merge_and_evaluate

CHECKS = [
    {"name": "multi_checkpoints", "command": "python /tmp/check_multi_ckpt.py"},
    {"name": "merge_metadata",    "command": f"python /tmp/check_merge.py {W}"},
    {"name": "merge_weights",     "command": check_merge_weights()},
]

task = merge_and_evaluate.task(
    prompt=(
        "Train multiple embedding model variants and merge them via SLERP for "
        "SciFact retrieval.\n"
        "You choose how many variants to train, what data to use, and the merge weights."
    ),
    graders=[
        {"name": "has_data",      "weight": 0.05, "command": f"ls {W}/data/*.jsonl 2>/dev/null | head -1"},
        *[{**c, "weight": 0.10} for c in CHECKS],
        {"name": "mteb_eval",     "weight": 0.10, "command": f"python /tmp/grader_eval.py {W}", "timeout": 1200},
        {"name": "ndcg@10>=0.10", "weight": 0.10, "command": "python /tmp/check_ndcg.py 0.10"},
        {"name": "ndcg@10>=0.30", "weight": 0.15, "command": "python /tmp/check_ndcg.py 0.30"},
        {"name": "ndcg@10>=0.50", "weight": 0.15, "command": "python /tmp/check_ndcg.py 0.50"},
        {"name": "ndcg@10>=0.65", "weight": 0.15, "command": "python /tmp/check_ndcg.py 0.65"},
    ],
)
task.slug = "merge_and_evaluate"
