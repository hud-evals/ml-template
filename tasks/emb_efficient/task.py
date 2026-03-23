from env import WORKSPACE as W, optimize_under_constraints
from tasks.graders import grader

task = optimize_under_constraints.task(
    prompt=(
        "Fine-tune Qwen3-0.6B into an embedding model for SciFact retrieval.\n\n"
        "CONSTRAINT: You have a strict compute budget of at most 200 total training steps "
        "across all training runs combined. Plan your training strategy carefully for "
        "maximum efficiency and write metrics.json into your best checkpoint directory."
    ),
    graders=[
        # TODO: Replace with a trace-backed budget grader once structured HUD step
        # data is available to local scenario evaluation.
        grader("check_step_budget", args=f"200 {W}", weight=0.25),
        grader("check_weights", args=W, weight=0.10),
        grader("eval", name="mteb_eval", args=f"emb mteb SciFact .emb_eval.json {W}", weight=0.10, timeout=1200),
        grader("check_threshold", name="ndcg@10>=0.50", args=f".emb_eval.json ndcg@10 0.50 {W}", weight=0.15),
        grader("check_threshold", name="ndcg@10>=0.70", args=f".emb_eval.json ndcg@10 0.70 {W}", weight=0.20),
        grader("check_threshold", name="ndcg@10>=0.85", args=f".emb_eval.json ndcg@10 0.85 {W}", weight=0.20),
    ],
    constraints={"max_total_steps": 200},
    setup_command=f"python /mcp_server/tasks/utils/setup_fixtures.py {W} --models Qwen/Qwen3-0.6B --datasets scifact",
)
task.slug = "emb_efficient"
