from env import WORKSPACE as W, repair_degraded_recipe
from tasks.graders import grader
from tasks.utils import load_patches

task = repair_degraded_recipe.task(
    prompt=(
        "Train a Flux diffusion model that converges well on CC12M data.\n"
        "After training completes, check the diagnostics output and verify "
        "the conditioning metrics look healthy."
    ),
    graders=[
        grader("check_code_fix", args=f"flux_causal_attn {W}", weight=0.40),
        grader("eval", args=f"flux .flux_eval.json {W}", weight=0.08, timeout=600),
        grader(
            "check_threshold",
            name="cond_uncond_cosine<=0.50",
            args=f".flux_eval.json cond_uncond_cosine 0.50 {W} lower",
            weight=0.52,
        ),
    ],
    patches=load_patches(__file__),
    staged_assets={
        "assets/cc12m": "data/cc12m",
    },
)
task.slug = "flux_debug_attn"
