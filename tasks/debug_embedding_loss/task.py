from env import WORKSPACE as W, check_weights, debug_embedding

task = debug_embedding.task(
    prompt="Fine-tune Qwen3-0.6B into an embedding model that performs well on SciFact retrieval.",
    graders=[
        {"name": "code_fix",          "weight": 0.15, "command": f"python /tmp/check_code_fix.py {W}"},
        {"name": "finetune_metadata", "weight": 0.15, "command": f"python /tmp/check_metadata.py finetune {W}"},
        {"name": "finetune_weights",  "weight": 0.15, "command": check_weights("finetune")},
        {"name": "mteb_eval",         "weight": 0.15, "command": f"python /tmp/grader_eval.py {W}", "timeout": 1200},
        {"name": "ndcg@10>=0.10",     "weight": 0.20, "command": f"python /tmp/check_ndcg.py 0.10 {W}"},
        {"name": "ndcg@10>=0.30",     "weight": 0.20, "command": f"python /tmp/check_ndcg.py 0.30 {W}"},
    ],
    mutation_type="buggy_loss",
)
task.slug = "debug_embedding_loss"
