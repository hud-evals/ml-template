from env import WORKSPACE as W, vlm_finetune

task = vlm_finetune.task(
    prompt="Train the VLM on CC12M image-caption data.",
    graders=[
        {"name": "vlm_metadata",   "weight": 0.15, "command": f"python /tmp/vlm_check_metadata.py {W}"},
        {"name": "vlm_checkpoint", "weight": 0.15, "command": f"python /tmp/vlm_check_checkpoint.py {W}"},
        {"name": "vlm_eval",       "weight": 0.20, "command": f"python /tmp/vlm_eval.py {W}", "timeout": 600},
        {"name": "val_loss<=5.0",  "weight": 0.15, "command": "python /tmp/vlm_check_loss.py 5.0"},
        {"name": "val_loss<=2.0",  "weight": 0.15, "command": "python /tmp/vlm_check_loss.py 2.0"},
        {"name": "val_loss<=1.0",  "weight": 0.20, "command": "python /tmp/vlm_check_loss.py 1.0"},
    ],
)
task.slug = "vlm_finetune"
