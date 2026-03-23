from env import WORKSPACE as W, repair_degraded_recipe
from tasks.graders import grader
from tasks.utils import load_patches

task = repair_degraded_recipe.task(
    prompt=(
        "Train the DeepSeek V3 debugmodel -- a Mixture-of-Experts language model.\n\n"
        "Your score depends on both validation loss and expert utilization balance."
    ),
    graders=[
        grader("check_code_fix", args=f"moe_load_balance {W}", weight=0.15),
        grader("eval", args=f"moe moe_debug_balance .moe_eval.json {W}", weight=0.10, timeout=600),
        grader("check_threshold", name="loss<=6.0", args=f".moe_eval.json loss 6.0 {W} lower", weight=0.10),
        grader("check_threshold", name="loss<=4.0", args=f".moe_eval.json loss 4.0 {W} lower", weight=0.20),
        grader("check_threshold", name="expert_balance>=0.6", args=f".moe_eval.json expert_balance 0.6 {W}", weight=0.15),
        grader("check_threshold", name="expert_balance>=0.85", args=f".moe_eval.json expert_balance 0.85 {W}", weight=0.30),
    ],
    patches=load_patches(__file__),
)
task.slug = "moe_debug_balance"
