from env import WORKSPACE as W, check_weights, data_audit_embedding

task = data_audit_embedding.task(
    prompt=(
        "Audit data/scifact.jsonl for quality issues, then train an embedding model "
        "for SciFact retrieval."
    ),
    graders=[
        {"name": "has_data",          "weight": 0.15, "command": f"ls {W}/data/*.jsonl 2>/dev/null | head -1"},
        {"name": "finetune_metadata", "weight": 0.15, "command": f"python /tmp/check_metadata.py finetune {W}"},
        {"name": "finetune_weights",  "weight": 0.10, "command": check_weights("finetune")},
        {"name": "mteb_eval",         "weight": 0.20, "command": f"python /tmp/grader_eval.py {W}", "timeout": 1200},
        {"name": "ndcg@10>=0.10",     "weight": 0.20, "command": "python /tmp/check_ndcg.py 0.10"},
        {"name": "ndcg@10>=0.30",     "weight": 0.20, "command": "python /tmp/check_ndcg.py 0.30"},
    ],
    contamination="data_leakage",
    leak_rate=0.2,
)
task.slug = "data_audit_leakage"
