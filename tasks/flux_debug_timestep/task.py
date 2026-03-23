from env import WORKSPACE as W, repair_degraded_recipe
from tasks.graders import grader
from tasks.utils import load_patches

task = repair_degraded_recipe.task(
    prompt=(
        "Train a Flux diffusion model that converges well on CC12M data.\n"
        "After training, verify the model can denoise properly by checking "
        "the diagnostics output."
    ),
    graders=[
        grader("check_code_fix", args=f"flux_zero_timestep {W}", weight=0.35),
        grader("eval", args=f"flux .flux_eval.json {W}", weight=0.05, timeout=600),
        grader(
            "check_threshold",
            name="cond_uncond_cosine<=0.40",
            args=f".flux_eval.json cond_uncond_cosine 0.40 {W} lower",
            weight=0.30,
        ),
        grader(
            "check_threshold",
            name="timestep_cosine<=0.40",
            args=f".flux_eval.json timestep_cosine 0.40 {W} lower",
            weight=0.30,
        ),
    ],
    patches=load_patches(__file__),
    setup_command=f"python /mcp_server/tasks/utils/setup_fixtures.py {W} --data-files pixparse/cc12m-wds cc12m-train-0000.tar cc12m-train-0001.tar cc12m-train-0002.tar",
)
task.slug = "flux_debug_timestep"
