from env import WORKSPACE as W, audit_evaluation_signal
from tasks.graders import grader

task = audit_evaluation_signal.task(
    prompt=(
        "Fine-tune Qwen3-0.6B into an embedding model that performs well on SciFact retrieval.\n"
        "Before relying on data/val.jsonl, verify that the evaluation signal is trustworthy and fix it if needed."
    ),
    graders=[
        grader("check_eval_cleaned", args=W, weight=0.20),
        grader("check_weights", args=W, weight=0.08),
        grader("check_weights", name="finetune_weights", args=f"finetune {W}", weight=0.08),
        grader("eval", name="mteb_eval", args=f"emb mteb SciFact .emb_eval.json {W}", weight=0.08, timeout=1200),
        grader("check_threshold", name="ndcg@10>=0.1596", args=f".emb_eval.json ndcg@10 0.1596 {W}", weight=0.16),
        grader("check_threshold", name="ndcg@10>=0.2092", args=f".emb_eval.json ndcg@10 0.2092 {W}", weight=0.18),
        grader("check_threshold", name="ndcg@10>=0.2587", args=f".emb_eval.json ndcg@10 0.2587 {W}", weight=0.22),
    ],
    eval_mutation="eval_leakage",
    leak_rate=0.25,
)
task.slug = "emb_audit_eval_leak"
