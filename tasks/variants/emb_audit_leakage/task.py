from env import WORKSPACE as W, audit_training_data
from tasks.graders import grader

task = audit_training_data.task(
    prompt=(
        "Audit data/scifact.jsonl for quality issues, then train an embedding model "
        "for SciFact retrieval."
    ),
    graders=[
        grader("check_data_cleaned", args=W, weight=0.16),
        grader("check_audit_provenance", args=W, weight=0.10),
        grader("check_weights", args=W, weight=0.06),
        grader("check_weights", name="finetune_weights", args=f"finetune {W}", weight=0.06),
        grader("eval", name="mteb_eval", args=f"emb mteb SciFact .emb_eval.json {W}", weight=0.08, timeout=1200),
        grader("check_threshold", name="ndcg@10>=0.1588", args=f".emb_eval.json ndcg@10 0.1588 {W}", weight=0.18),
        grader("check_threshold", name="ndcg@10>=0.1843", args=f".emb_eval.json ndcg@10 0.1843 {W}", weight=0.18),
        grader("check_threshold", name="ndcg@10>=0.2099", args=f".emb_eval.json ndcg@10 0.2099 {W}", weight=0.18),
    ],
    contamination="data_leakage",
    leak_rate=0.2,
)
task.slug = "emb_audit_leakage"
