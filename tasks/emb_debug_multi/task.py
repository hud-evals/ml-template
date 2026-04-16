from env import WORKSPACE as W, repair_degraded_recipe
from tasks.graders import grader
from tasks.utils import load_patches

task = repair_degraded_recipe.task(
    prompt="Fine-tune Qwen3-0.6B into an embedding model that performs well on SciFact retrieval.",
    graders=[
        grader("check_code_fix", name="code_fix_loss", args=f"buggy_loss {W}", weight=0.13),
        grader("check_code_fix", name="code_fix_pooling", args=f"bad_pooling {W}", weight=0.12),
        grader("check_weights", args=W, weight=0.20),
        grader("eval", name="mteb_eval", args=f"emb mteb SciFact .emb_eval.json {W}", weight=0.15, timeout=1200),
        grader("check_threshold", name="ndcg@10>=0.10", args=f".emb_eval.json ndcg@10 0.10 {W}", weight=0.20),
        grader("check_threshold", name="ndcg@10>=0.30", args=f".emb_eval.json ndcg@10 0.30 {W}", weight=0.20),
    ],
    patches=load_patches(__file__),
    setup_command=f"python /mcp_server/setup/setup_fixtures.py {W} --models Qwen/Qwen3-0.6B --datasets scifact synthetic",
)
task.slug = "emb_debug_multi"
