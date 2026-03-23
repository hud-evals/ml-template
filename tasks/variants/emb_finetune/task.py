from env import WORKSPACE as W, train_to_target
from tasks.graders import grader

task = train_to_target.task(
    prompt=(
        "Fine-tune Qwen3-0.6B into an embedding model that performs well on SciFact retrieval.\n"
        "You choose the hyperparameters and data strategy."
    ),
    graders=[
        {"name": "has_data", "command": f"ls {W}/data/*.jsonl 2>/dev/null | head -1", "weight": 0.04},
        grader("check_weights", args=W, weight=0.06),
        grader("check_weights", name="finetune_weights", args=f"finetune {W}", weight=0.06),
        grader("eval", name="mteb_eval", args=f"emb mteb SciFact .emb_eval.json {W}", weight=0.08, timeout=1200),
        grader("check_threshold", name="ndcg@10>=0.1596", args=f".emb_eval.json ndcg@10 0.1596 {W}", weight=0.16),
        grader("check_threshold", name="ndcg@10>=0.2092", args=f".emb_eval.json ndcg@10 0.2092 {W}", weight=0.18),
        grader("check_threshold", name="ndcg@10>=0.2587", args=f".emb_eval.json ndcg@10 0.2587 {W}", weight=0.20),
        grader("check_threshold", name="ndcg@10>=0.3083", args=f".emb_eval.json ndcg@10 0.3083 {W}", weight=0.22),
    ],
)
task.slug = "emb_finetune"
