from env import WORKSPACE as W, multistage_retrieval
from tasks.finetune_embedding.task import CHECKS as FINETUNE_CHECKS
from tasks.merge_and_evaluate.task import CHECKS as MERGE_CHECKS
from tasks.pretrain_embedding.task import CHECKS as PRETRAIN_CHECKS

task = multistage_retrieval.task(
    prompt=(
        "Build an embedding model for SciFact retrieval using a multi-stage pipeline:\n"
        "  1. Pre-train on synthetic data\n"
        "  2. Fine-tune on labeled data\n"
        "  3. Merge checkpoints via SLERP\n\n"
        "You choose the hyperparameters, data strategy, and configuration for each stage."
    ),
    graders=[
        *[{**c, "weight": 0.05} for c in PRETRAIN_CHECKS],
        *[{**c, "weight": 0.05} for c in FINETUNE_CHECKS],
        {"name": "finetune_resumed", "weight": 0.05, "command": f"python /tmp/check_resume.py {W}"},
        *[{**c, "weight": 0.05} for c in MERGE_CHECKS],
        {"name": "mteb_eval",     "weight": 0.10, "command": f"python /tmp/grader_eval.py {W}", "timeout": 1200},
        {"name": "ndcg@10>=0.10", "weight": 0.10, "command": f"python /tmp/check_ndcg.py 0.10 {W}"},
        {"name": "ndcg@10>=0.30", "weight": 0.10, "command": f"python /tmp/check_ndcg.py 0.30 {W}"},
        {"name": "ndcg@10>=0.50", "weight": 0.15, "command": f"python /tmp/check_ndcg.py 0.50 {W}"},
        {"name": "ndcg@10>=0.65", "weight": 0.15, "command": f"python /tmp/check_ndcg.py 0.65 {W}"},
    ],
)
task.slug = "multistage_retrieval"
