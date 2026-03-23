from env import WORKSPACE as W, train_to_target
from tasks.graders import grader

task = train_to_target.task(
    prompt="Train the Qwen3+SigLIP2 VLM on CC12M image-caption data using the qwen3_0.6B flavor.",
    graders=[
        grader("check_weights", args=W, weight=0.08),
        grader("check_checkpoint", args=W, weight=0.08),
        grader("eval", args=f"vlm .vlm_eval.json {W}", weight=0.12, timeout=600),
        grader("check_threshold", name="val_loss<=0.0555", args=f".vlm_eval.json val_loss 0.0555 {W} lower", weight=0.18),
        grader("check_threshold", name="val_loss<=0.0398", args=f".vlm_eval.json val_loss 0.0398 {W} lower", weight=0.24),
        grader("check_threshold", name="val_loss<=0.0242", args=f".vlm_eval.json val_loss 0.0242 {W} lower", weight=0.30),
    ],
)
task.slug = "vlm_finetune"
