from env import WORKSPACE as W, compose_multi_stage_pipeline
from tasks.graders import grader

task = compose_multi_stage_pipeline.task(
    prompt=(
        "Train multiple embedding model variants and merge them via SLERP to maximize "
        "held-out retrieval quality across SciFact and Natural Questions.\n"
        "You choose how many variants to train, what data to use, and the merge weights."
    ),
    graders=[
        {"name": "has_data", "command": f"ls {W}/data/*.jsonl 2>/dev/null | head -1", "weight": 0.03},
        grader("check_checkpoint", name="check_multi_ckpt", args=f"{W} 2 no-merged", weight=0.06),
        grader("check_merge", args=W, weight=0.06),
        grader("check_weights", name="merge_weights", args=f"merged {W}", weight=0.06),
        grader("check_merge_gain", args=f"0.03 {W}", weight=0.16),
        grader("eval", name="mteb_eval", args=f"emb local data/merge_val.jsonl .emb_eval.json {W}", weight=0.07, timeout=1200),
        grader("check_threshold", name="ndcg@10>=0.22", args=f".emb_eval.json ndcg@10 0.22 {W}", weight=0.14),
        grader("check_threshold", name="ndcg@10>=0.36", args=f".emb_eval.json ndcg@10 0.36 {W}", weight=0.14),
        grader("check_threshold", name="ndcg@10>=0.50", args=f".emb_eval.json ndcg@10 0.50 {W}", weight=0.14),
        grader("check_threshold", name="ndcg@10>=0.62", args=f".emb_eval.json ndcg@10 0.62 {W}", weight=0.14),
    ],
    expected_stages=["train_variant", "merge"],
)
task.slug = "emb_merge_eval"
