import json
import subprocess
import sys
from pathlib import Path

from ..conftest import GRADERS_DIR, make_workspace


class TestFluxEval:
    def test_flux_eval_collects_best_metrics(self, tmp_path):
        ws = Path(make_workspace(str(tmp_path)))

        good = ws / "checkpoints" / "good"
        good.mkdir(parents=True)
        (good / "diagnostics.json").write_text(
            json.dumps(
                {
                    "cond_uncond_cosine": 0.48,
                    "cond_uncond_mse": 0.12,
                    "output_variance": 0.20,
                }
            )
        )
        (good / "train.log").write_text("global_avg_loss: 1.08\n")

        better_loss = ws / "checkpoints" / "better_loss"
        better_loss.mkdir(parents=True)
        (better_loss / "diagnostics.json").write_text(
            json.dumps(
                {
                    "cond_uncond_cosine": 0.61,
                    "cond_uncond_mse": 0.05,
                    "output_variance": 0.11,
                }
            )
        )
        (better_loss / "train.log").write_text("step 200: loss=0.94 grad_norm=0.1\n")

        result = subprocess.run(
            [sys.executable, str(GRADERS_DIR / "eval.py"), "flux", ".flux_eval.json", str(ws)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        cache = json.loads((ws / ".flux_eval.json").read_text())
        assert cache["cond_uncond_cosine"] == 0.48
        assert cache["cond_uncond_mse"] == 0.12
        assert cache["last_loss"] == 0.94
        assert len(cache["candidates"]) == 2
