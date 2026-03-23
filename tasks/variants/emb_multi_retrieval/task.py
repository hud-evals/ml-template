from env import WORKSPACE as W, compose_multi_stage_pipeline
from tasks.graders import grader

task = compose_multi_stage_pipeline.task(
    prompt=(
        "Build an embedding model for SciFact retrieval using a multi-stage pipeline:\n"
        "  1. Pre-train on synthetic data\n"
        "  2. Fine-tune on labeled data\n"
        "  3. Merge checkpoints via SLERP\n\n"
        "You choose the hyperparameters, data strategy, and configuration for each stage."
    ),
    graders=[
        {"name": "has_data", "command": f"ls {W}/data/*.jsonl 2>/dev/null | head -1", "weight": 0.03},
        grader("check_weights", args=W, weight=0.05),
        grader("check_weights", name="pretrain_weights", args=f"pretrain {W}", weight=0.04),
        grader("check_weights", args=W, weight=0.04),
        grader("check_weights", name="finetune_weights", args=f"finetune {W}", weight=0.05),
        grader("check_weights", args=W, weight=0.06),
        grader("check_checkpoint", name="check_multi_ckpt", args=f"{W} 2 no-merged", weight=0.04),
        grader("check_merge", args=W, weight=0.04),
        grader("check_weights", name="merge_weights", args=f"merged {W}", weight=0.04),
        grader("check_merge_gain", args=f"0.03 {W}", weight=0.12),
        grader("eval", name="mteb_eval", args=f"emb mteb SciFact .emb_eval.json {W}", weight=0.06, timeout=1200),
        grader("check_threshold", name="ndcg@10>=0.4524", args=f".emb_eval.json ndcg@10 0.4524 {W}", weight=0.11),
        grader("check_threshold", name="ndcg@10>=0.4697", args=f".emb_eval.json ndcg@10 0.4697 {W}", weight=0.11),
        grader("check_threshold", name="ndcg@10>=0.4871", args=f".emb_eval.json ndcg@10 0.4871 {W}", weight=0.11),
        grader("check_threshold", name="ndcg@10>=0.5044", args=f".emb_eval.json ndcg@10 0.5044 {W}", weight=0.10),
    ],
    expected_stages=["pretrain", "finetune", "merge"],
)
task.slug = "emb_multi_retrieval"
