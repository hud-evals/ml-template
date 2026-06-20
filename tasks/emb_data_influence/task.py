from env import WORKSPACE as W, targeted_failure_recovery
from tasks.graders import grader

task = targeted_failure_recovery(
    prompt=(
        "A model was trained on data/scifact.jsonl and a checkpoint is at "
        "checkpoints/checkpoint_61/. The model performs reasonably overall but fails "
        "on specific queries listed in failing_queries.json.\n\n"
        "Find the root cause and fix it. You must resume from checkpoints/checkpoint_61/. "
        "Your score depends on overall nDCG and on nDCG for the previously-failing queries."
    ),
    graders=[
        grader("check_weights", args=W, weight=0.20),
        grader("eval", name="mteb_eval", args=f"emb mteb SciFact .emb_eval.json {W}", weight=0.05, timeout=1200),
        grader("check_threshold", name="overall_ndcg>=0.40", args=f".emb_eval.json ndcg@10 0.40 {W}", weight=0.10),
        grader(
            "eval",
            name="failing_query_eval",
            args=f"emb local data/failing_queries_val.jsonl .failing_query_eval.json {W}",
            weight=0.05,
            timeout=1200,
        ),
        grader(
            "check_threshold",
            name="failing_ndcg>=0.10",
            args=f".failing_query_eval.json ndcg@10 0.10 {W}",
            weight=0.15,
        ),
        grader(
            "check_threshold",
            name="failing_ndcg>=0.20",
            args=f".failing_query_eval.json ndcg@10 0.20 {W}",
            weight=0.20,
        ),
        grader(
            "check_threshold",
            name="failing_ndcg>=0.30",
            args=f".failing_query_eval.json ndcg@10 0.30 {W}",
            weight=0.25,
        ),
    ],
    failure_manifest={
        "required_resume_from": "checkpoints/checkpoint_61",
        "failure_subset_file": "failing_queries.json",
        "objective_keys": ["ndcg@10", "failing_query_ndcg@10"],
        "artifacts": [
            "failing_queries.json",
            "data/failing_queries_val.jsonl",
            "checkpoints/checkpoint_61",
        ],
    },
    setup_command=(
        f"python /mcp_server/tasks/utils/setup_fixtures.py {W} --models Qwen/Qwen3-0.6B --datasets scifact && "
        f"python /mcp_server/tasks/emb_data_influence/setup.py {W}"
    ),
)
task.slug = "emb_data_influence"
