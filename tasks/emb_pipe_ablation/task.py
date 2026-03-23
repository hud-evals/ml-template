from env import WORKSPACE as W, targeted_failure_recovery
from tasks.graders import grader

task = targeted_failure_recovery.task(
    prompt=(
        "A 3-stage training pipeline produced a disappointing result "
        "(see pipeline_result.json). Intermediate checkpoints are at "
        "checkpoints/stage1_pretrain/, checkpoints/stage2_finetune/, "
        "and checkpoints/stage3_merged/.\n\n"
        "Diagnose the problem and produce a model with nDCG@10 >= 0.55 on SciFact.\n\n"
        "Constraint: You have a budget of 100 additional training steps."
    ),
    graders=[
        grader("check_step_budget", name="step_budget", args=f"100 {W}", weight=0.15),
        grader("check_weights", args=W, weight=0.20),
        grader("eval", name="mteb_eval", args=f"emb mteb SciFact .emb_eval.json {W}", weight=0.05, timeout=1200),
        grader("check_threshold", name="ndcg@10>=0.35", args=f".emb_eval.json ndcg@10 0.35 {W}", weight=0.12),
        grader("check_threshold", name="ndcg@10>=0.45", args=f".emb_eval.json ndcg@10 0.45 {W}", weight=0.13),
        grader("check_threshold", name="ndcg@10>=0.55", args=f".emb_eval.json ndcg@10 0.55 {W}", weight=0.15),
        grader("check_threshold", name="ndcg@10>=0.65", args=f".emb_eval.json ndcg@10 0.65 {W}", weight=0.20),
    ],
    failure_manifest={
        "required_resume_from": "checkpoints/stage",
        "max_extra_steps": 100,
        "objective_keys": ["ndcg@10"],
        "artifacts": [
            "pipeline_result.json",
            "checkpoints/stage1_pretrain",
            "checkpoints/stage2_finetune",
            "checkpoints/stage3_merged",
        ],
    },
    setup_command=(
        f"python /mcp_server/tasks/utils/setup_fixtures.py {W} --models Qwen/Qwen3-0.6B --datasets scifact synthetic && "
        f"python /mcp_server/tasks/emb_pipe_ablation/setup.py {W}"
    ),
)
task.slug = "emb_pipe_ablation"
