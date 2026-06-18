from env import WORKSPACE as W, certify_reliability
from tasks.graders import grader
from tasks.utils import load_patches

task = certify_reliability(
    prompt=(
        "Run a deterministic seed sweep of GPT-OSS debugmodel on a single GPU. "
        "Use seeds 0, 42, 123, and 999, training each for a few steps with "
        "--debug.deterministic enabled, and confirm that every seed converges cleanly."
    ),
    graders=[
        grader("eval", args=f"moe moe_seed_sweep .moe_eval.json {W}", weight=0.22, timeout=600),
        grader("check_threshold", name="pass_rate>=0.34", args=f".moe_eval.json pass_rate 0.34 {W}", weight=0.24),
        grader("check_threshold", name="pass_rate>=0.67", args=f".moe_eval.json pass_rate 0.67 {W}", weight=0.24),
        grader("check_threshold", name="pass_rate>=1.0", args=f".moe_eval.json pass_rate 1.0 {W}", weight=0.30),
    ],
    patches=load_patches(__file__),
    reliability_matrix=[
        {
            "name": f"seed_{seed}",
            "env": {"NGPU": "1", "MODULE": "gpt_oss", "CONFIG": "gpt_oss_debugmodel"},
            "args": [f"--debug.seed={seed}", "--debug.deterministic", "--training.steps=5"],
        }
        for seed in (0, 42, 123, 999)
    ],
)
task.slug = "moe_seed_sweep"
