"""Model merging via Spherical Linear Interpolation (SLERP)."""

import argparse
import json
import logging
import os

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _slerp(t: float, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    v0_flat = v0.float().flatten()
    v1_flat = v1.float().flatten()

    v0_norm = torch.nn.functional.normalize(v0_flat, dim=0)
    v1_norm = torch.nn.functional.normalize(v1_flat, dim=0)

    dot = torch.clamp(torch.dot(v0_norm, v1_norm), -1.0, 1.0)
    omega = torch.acos(dot)

    if omega.abs() < 1e-6:
        result = (1.0 - t) * v0_flat + t * v1_flat
    else:
        sin_omega = torch.sin(omega)
        result = (torch.sin((1.0 - t) * omega) / sin_omega) * v0_flat + \
                 (torch.sin(t * omega) / sin_omega) * v1_flat

    return result.reshape(v0.shape).to(v0.dtype)


def slerp_merge(
    checkpoints: list[str],
    output_dir: str,
    weights: list[float] | None = None,
):
    if len(checkpoints) < 2:
        raise ValueError("Need at least 2 checkpoints to merge")

    if weights is None:
        weights = [1.0 / len(checkpoints)] * len(checkpoints)

    if len(weights) != len(checkpoints):
        raise ValueError(f"Got {len(weights)} weights for {len(checkpoints)} checkpoints")

    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]

    logger.info("Loading %d checkpoints for SLERP merge", len(checkpoints))
    state_dicts = []
    for ckpt in checkpoints:
        from transformers import AutoModel
        m = AutoModel.from_pretrained(ckpt, trust_remote_code=True)
        state_dicts.append(m.state_dict())
        del m

    merged_sd = state_dicts[0]
    cumulative_weight = weights[0]

    for i in range(1, len(state_dicts)):
        t = weights[i] / (cumulative_weight + weights[i])
        logger.info("SLERP merge: checkpoint %d/%d (t=%.3f)", i + 1, len(state_dicts), t)

        for key in merged_sd:
            if merged_sd[key].dtype in (torch.float16, torch.bfloat16, torch.float32):
                merged_sd[key] = _slerp(t, merged_sd[key], state_dicts[i][key])
            else:
                merged_sd[key] = state_dicts[i][key]

        cumulative_weight += weights[i]

    os.makedirs(output_dir, exist_ok=True)

    from transformers import AutoModel, AutoTokenizer
    base_model = AutoModel.from_pretrained(checkpoints[0], trust_remote_code=True)
    base_model.load_state_dict(merged_sd)
    base_model.save_pretrained(output_dir)
    del base_model

    tokenizer = AutoTokenizer.from_pretrained(checkpoints[0], trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "method": "slerp",
        "source_checkpoints": checkpoints,
        "weights": weights,
    }
    with open(os.path.join(output_dir, "merge_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Merged model saved to %s", output_dir)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Merge model checkpoints via SLERP")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="*", default=None)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    slerp_merge(args.checkpoints, args.output_dir, args.weights)


if __name__ == "__main__":
    main()
