from env import WORKSPACE as W, train_to_target
from tasks.graders import grader

task = train_to_target.task(
    prompt=(
        "Pre-train Qwen3-0.6B on synthetic data as the first stage of an embedding pipeline.\n"
        "Optimize for SciFact retrieval quality as measured by the MTEB SciFact nDCG@10 grader.\n"
        "You choose the hyperparameters."
    ),
    graders=[
        {"name": "has_data", "command": f"ls {W}/data/*.jsonl 2>/dev/null | head -1", "weight": 0.05},
        grader("check_weights", args=W, weight=0.10),
        grader("check_weights", name="pretrain_weights", args=f"pretrain {W}", weight=0.10),
        grader("eval", name="mteb_eval", args=f"emb mteb SciFact .emb_eval.json {W}", weight=0.10, timeout=1200),
        grader("check_threshold", name="ndcg@10>=0.2477", args=f".emb_eval.json ndcg@10 0.2477 {W}", weight=0.20),
        grader("check_threshold", name="ndcg@10>=0.2524", args=f".emb_eval.json ndcg@10 0.2524 {W}", weight=0.20),
        grader("check_threshold", name="ndcg@10>=0.2572", args=f".emb_eval.json ndcg@10 0.2572 {W}", weight=0.25),
    ],
)
task.slug = "emb_pretrain"
