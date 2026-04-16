from env import WORKSPACE as W, repair_degraded_recipe
from tasks.graders import grader
from tasks.utils import load_patches

task = repair_degraded_recipe.task(
    prompt="Train the Qwen3+SigLIP2 VLM on CC12M image-caption data using the qwen3_0.6B flavor.",
    graders=[
        grader("check_code_fix", args=f"buggy_projector {W}", weight=0.16),
        grader("check_checkpoint", args=W, weight=0.08),
        grader("eval", args=f"vlm .vlm_eval.json {W}", weight=0.12, timeout=600),
        grader("check_threshold", name="val_loss<=0.06", args=f".vlm_eval.json val_loss 0.06 {W} lower", weight=0.18),
        grader("check_threshold", name="val_loss<=0.03", args=f".vlm_eval.json val_loss 0.03 {W} lower", weight=0.22),
        grader("check_threshold", name="val_loss<=0.01", args=f".vlm_eval.json val_loss 0.01 {W} lower", weight=0.24),
    ],
    patches=load_patches(__file__),
    setup_command=f"python /mcp_server/setup/setup_fixtures.py {W} --models Qwen/Qwen3-0.6B --data-files pixparse/cc12m-wds cc12m-train-0000.tar cc12m-train-0001.tar cc12m-train-0002.tar",
)
task.slug = "vlm_debug_projector"
