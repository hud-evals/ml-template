from env import WORKSPACE as W, repair_degraded_recipe
from tasks.graders import grader
from tasks.utils import load_patches

task = repair_degraded_recipe.task(
    prompt="Fine-tune Qwen3-0.6B into an embedding model that performs well on SciFact retrieval.",
    graders=[
        grader("check_code_fix", args=f"bad_pooling {W}", weight=0.12),
        grader("check_weights", args=W, weight=0.08),
        grader("check_weights", name="finetune_weights", args=f"finetune {W}", weight=0.08),
        grader("eval", name="mteb_eval", args=f"emb mteb SciFact .emb_eval.json {W}", weight=0.08, timeout=1200),
        grader("check_threshold", name="ndcg@10>=0.172", args=f".emb_eval.json ndcg@10 0.172 {W}", weight=0.32),
        grader("check_threshold", name="ndcg@10>=0.1838", args=f".emb_eval.json ndcg@10 0.1838 {W}", weight=0.32),
    ],
    patches=load_patches(__file__),
)
task.slug = "emb_debug_pooling"
