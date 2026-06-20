from env import WORKSPACE as W, audit_training_data
from tasks.graders import grader

task = audit_training_data(
    prompt=(
        "Audit data/scifact.jsonl for quality issues. Clean the data in place, "
        "then train an embedding model for SciFact retrieval."
    ),
    graders=[
        grader("check_data_cleaned", args=W, weight=0.15),
        grader("check_audit_provenance", args=W, weight=0.10),
        grader("check_weights", args=W, weight=0.20),
        grader("eval", name="mteb_eval", args=f"emb mteb SciFact .emb_eval.json {W}", weight=0.15, timeout=1200),
        grader("check_threshold", name="ndcg@10>=0.16", args=f".emb_eval.json ndcg@10 0.16 {W}", weight=0.20),
        grader("check_threshold", name="ndcg@10>=0.18", args=f".emb_eval.json ndcg@10 0.18 {W}", weight=0.20),
    ],
    contamination="label_noise",
    noise_rate=0.3,
    setup_command=f"python /mcp_server/tasks/utils/setup_fixtures.py {W} --models Qwen/Qwen3-0.6B --datasets scifact",
)
task.slug = "emb_audit_noise"
