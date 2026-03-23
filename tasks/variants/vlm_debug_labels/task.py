from env import WORKSPACE as W, repair_degraded_recipe
from tasks.graders import grader
from tasks.utils import load_patches

task = repair_degraded_recipe.task(
    prompt="Train the Qwen3+SigLIP2 VLM on CC12M image-caption data using the qwen3_0.6B flavor.",
    graders=[
        grader("check_code_fix", args=f"bad_label_mask {W}", weight=0.16),
        grader("check_weights", args=W, weight=0.08),
        grader("check_checkpoint", args=W, weight=0.08),
        grader("eval", args=f"vlm .vlm_eval.json {W}", weight=0.12, timeout=600),
        grader("check_threshold", name="val_loss<=0.064", args=f".vlm_eval.json val_loss 0.064 {W} lower", weight=0.24),
        grader("check_threshold", name="val_loss<=0.0606", args=f".vlm_eval.json val_loss 0.0606 {W} lower", weight=0.32),
    ],
    patches=load_patches(__file__),
)
task.slug = "vlm_debug_labels"
