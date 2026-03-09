from env import WORKSPACE as W, vlm_finetune

task = vlm_finetune.task(
    prompt="Train the Qwen3+SigLIP2 VLM on CC12M image-caption data using the qwen3_0.6B flavor.",
    graders=[
        {"name": "vlm_metadata",   "weight": 0.10, "command": f"python /tmp/vlm_check_metadata.py {W}"},
        {"name": "vlm_checkpoint", "weight": 0.10, "command": f"python /tmp/vlm_check_checkpoint.py {W}"},
        {"name": "vlm_eval",       "weight": 0.20, "command": f"python /tmp/vlm_eval.py {W}", "timeout": 600},
        {"name": "val_loss<=8.0",  "weight": 0.15, "command": f"python /tmp/vlm_check_loss.py 8.0 {W}"},
        {"name": "val_loss<=5.0",  "weight": 0.20, "command": f"python /tmp/vlm_check_loss.py 5.0 {W}"},
        {"name": "val_loss<=3.0",  "weight": 0.25, "command": f"python /tmp/vlm_check_loss.py 3.0 {W}"},
    ],
)
task.slug = "vlm_finetune"
