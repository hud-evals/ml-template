from env import WORKSPACE as W, repair_degraded_recipe
from tasks.graders import grader
from tasks.utils import load_patches

task = repair_degraded_recipe.task(
    prompt="Fine-tune Qwen3-0.6B into an embedding model that performs well on SciFact retrieval.",
    graders=[
        grader("check_code_fix", args=f"fp16_logits {W}", weight=0.30),
        grader("check_weights", args=W, weight=0.10),
        grader("check_weights", name="finetune_weights", args=f"finetune {W}", weight=0.10),
        grader("eval", name="mteb_eval", args=f"emb mteb SciFact .emb_eval.json {W}", weight=0.15, timeout=1200),
        grader("check_threshold", name="ndcg@10>=0.10", args=f".emb_eval.json ndcg@10 0.10 {W}", weight=0.15),
        grader("check_threshold", name="ndcg@10>=0.30", args=f".emb_eval.json ndcg@10 0.30 {W}", weight=0.20),
    ],
    patches=load_patches(__file__),
)
task.slug = "emb_debug_precision"
