from env import WORKSPACE as W, adapt_without_forgetting
from tasks.graders import grader

task = adapt_without_forgetting.task(
    prompt=(
        "You have a model at checkpoints/scifact_base/ that performs well on "
        "SciFact retrieval. Adapt it to also handle Natural Questions (NQ).\n\n"
        "Constraints:\n"
        "  - SciFact training data has been deleted. You cannot retrain on SciFact.\n"
        "  - You have data/nq.jsonl for NQ training and data/nq_val.jsonl for evaluation.\n"
        "  - data/val.jsonl is the SciFact validation set.\n"
        "  - You must resume from checkpoints/scifact_base/.\n\n"
        "Your model will be evaluated on both SciFact and NQ."
    ),
    graders=[
        grader("check_weights", args=W, weight=0.10),
        grader("eval", name="scifact_eval", args=f"emb mteb SciFact .emb_eval.json {W}", weight=0.05, timeout=1200),
        grader("check_threshold", name="scifact>=0.45", args=f".emb_eval.json ndcg@10 0.45 {W}", weight=0.10),
        grader("check_threshold", name="scifact>=0.55", args=f".emb_eval.json ndcg@10 0.55 {W}", weight=0.15),
        grader("eval", name="nq_eval", args=f"emb local /mcp_server/data/nq_grader_val.jsonl .nq_eval.json {W}", weight=0.05, timeout=1200),
        grader("check_threshold", name="nq>=0.70", args=f".nq_eval.json ndcg@10 0.70 {W}", weight=0.10),
        grader("check_threshold", name="nq>=0.80", args=f".nq_eval.json ndcg@10 0.80 {W}", weight=0.20),
        grader("check_threshold", name="nq>=0.90", args=f".nq_eval.json ndcg@10 0.90 {W}", weight=0.25),
    ],
    base_checkpoint="checkpoints/scifact_base",
    adapt_train_files=["data/nq.jsonl"],
    retain_eval_files=["data/val.jsonl", "data/nq_val.jsonl"],
    forbidden_train_files=["data/scifact.jsonl"],
    setup_command=f"python /mcp_server/setup/setup_fixtures.py {W} --models Qwen/Qwen3-0.6B --datasets scifact nq --checkpoints scifact_base",
)
task.slug = "emb_adapt_nq"
