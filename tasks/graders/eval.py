"""Unified evaluation grader for all model types.

Discovers checkpoints (DCP or HF format), evaluates them, and caches results.

Usage:
  eval.py emb mteb <tasks_csv> <cache_file> <workspace> [max_seq_length]
  eval.py emb local <eval_data_relpath> <cache_file> <workspace> [max_seq_length]
  eval.py vlm <cache_file> <workspace>
  eval.py flux <cache_file> <workspace>
  eval.py moe <task_slug> <cache_file> <workspace>
"""

import glob
import json
import os
import re
import sys

_EXCLUDE = ("assets/", "tests/", ".venv/")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# ---------------------------------------------------------------------------
# Checkpoint discovery (shared across all model types)
# ---------------------------------------------------------------------------


def find_checkpoints(workspace: str) -> tuple[list[str], list[str]]:
    """Find all checkpoints in workspace.

    Returns (hf_dirs, dcp_dirs) where:
      hf_dirs: directories containing model.safetensors
      dcp_dirs: step-N directories containing .metadata (DCP format)
    """
    hf_dirs = []
    dcp_dirs = []

    for pattern in ["**/model.safetensors", "**/model-*.safetensors", "**/model.safetensors.index.json", "**/pytorch_model.bin"]:
        for wf in glob.glob(f"{workspace}/{pattern}", recursive=True):
            rel = os.path.relpath(wf, workspace)
            if any(rel.startswith(p) for p in _EXCLUDE):
                continue
            hf_dirs.append(os.path.dirname(wf))

    for mf in glob.glob(f"{workspace}/**/.metadata", recursive=True):
        d = os.path.dirname(mf)
        rel = os.path.relpath(d, workspace)
        if any(rel.startswith(p) for p in _EXCLUDE):
            continue
        if os.path.basename(d).startswith("step-"):
            dcp_dirs.append(d)

    return sorted(set(hf_dirs)), sorted(set(dcp_dirs))


def dcp_to_hf(dcp_dir: str, workspace: str) -> str:
    """Convert a DCP checkpoint to HF format using torchtitan's convert_to_hf. Returns HF dir path."""
    import subprocess

    hf_dir = dcp_dir + "_hf"
    if os.path.exists(os.path.join(hf_dir, "model.safetensors")):
        return hf_dir

    hf_assets = os.path.join(workspace, "assets", "hf", "Qwen3-0.6B")
    if not os.path.isdir(hf_assets):
        hf_assets = os.path.join(workspace, "tests", "assets", "tokenizer")

    for base in [src_dir, workspace]:
        script = os.path.join(base, "scripts", "checkpoint_conversion", "convert_to_hf.py")
        if os.path.isfile(script):
            # dcp.load/save need distributed, so run via torchrun
            result = subprocess.run(
                [
                    "torchrun", "--nproc_per_node", "1",
                    "--rdzv_backend", "c10d", "--rdzv_endpoint", "localhost:0",
                    script, dcp_dir, hf_dir,
                    "--model_name", "qwen3", "--model_flavor", "0.6B",
                    "--hf_assets_path", hf_assets,
                ],
                env={**os.environ, "PYTHONPATH": src_dir},
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                return hf_dir
            raise RuntimeError(f"DCP conversion failed:\n{result.stderr[-2000:]}")

    raise FileNotFoundError("convert_to_hf.py not found")

    print(f"Converted DCP {dcp_dir} -> HF {hf_dir}")
    return hf_dir


# ---------------------------------------------------------------------------
# Embedding evaluation
# ---------------------------------------------------------------------------


def eval_emb(args: list[str]) -> None:
    mode = args[0]
    if mode == "mteb":
        target, cache_file, workspace = args[1], args[2], args[3]
        max_seq_length = int(args[4]) if len(args) > 4 else 512
        eval_plan = {"kind": "mteb", "tasks": target.split(","), "max_seq_length": max_seq_length}
    elif mode == "local":
        target, cache_file, workspace = args[1], args[2], args[3]
        max_seq_length = int(args[4]) if len(args) > 4 else 512
        eval_plan = {"kind": "local", "eval_data": target, "max_seq_length": max_seq_length}
    else:
        print(f"Unknown emb mode: {mode}")
        sys.exit(1)

    src_dir = os.environ.get("SRC_DIR", "/mcp_server")
    cache_path = os.path.join(workspace, cache_file)

    hf_dirs, dcp_dirs = find_checkpoints(workspace)

    # Convert DCP to HF if no HF checkpoints
    if not hf_dirs:
        for d in dcp_dirs:
            try:
                hf_dirs.append(dcp_to_hf(d, workspace))
            except Exception as e:
                print(f"DCP conversion failed for {d}: {e}")

    if not hf_dirs:
        print("No model checkpoints found")
        sys.exit(1)

    hf_dirs = sorted(hf_dirs, key=lambda d: os.path.getmtime(d), reverse=True)

    sys.path.insert(0, src_dir)
    from torchtitan.experiments.embedding.evaluate import evaluate_local, evaluate_mteb

    best_ndcg = -1.0
    best_dir = None
    candidates = []
    for idx, d in enumerate(hf_dirs[:10]):
        try:
            if eval_plan["kind"] == "local":
                metrics = evaluate_local(d, os.path.join(workspace, eval_plan["eval_data"]), max_seq_length=eval_plan["max_seq_length"])
            else:
                metrics = evaluate_mteb(d, eval_plan["tasks"], max_seq_length=eval_plan["max_seq_length"])
            ndcg = metrics.get("ndcg@10", 0)
            candidates.append({"checkpoint_dir": d, "ndcg@10": ndcg, "eval_kind": eval_plan["kind"]})
            print(f"[{idx + 1}/{min(len(hf_dirs), 10)}] {d}: nDCG@10={ndcg:.4f}")
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_dir = d
            if best_ndcg >= 0.65:
                print("Above golden threshold, stopping early")
                break
        except Exception as e:
            print(f"{d}: eval failed ({e})")

    result = {"ndcg@10": best_ndcg, "best_dir": best_dir, "candidates": candidates}
    with open(cache_path, "w") as f:
        json.dump(result, f)
    print(f"Best: {best_dir} nDCG@10={best_ndcg:.4f}")


# ---------------------------------------------------------------------------
# VLM evaluation
# ---------------------------------------------------------------------------


def eval_vlm(args: list[str]) -> None:
    cache_file, workspace = args[0], args[1]
    src_dir = os.environ.get("SRC_DIR", "/mcp_server")
    cache_path = os.path.join(workspace, cache_file)

    hf_dirs, dcp_dirs = find_checkpoints(workspace)

    # VLM uses DCP checkpoints with torchtitan's evaluate_vlm
    # Find the dump_folder (grandparent of step-N, since evaluate_vlm
    # sets dump_folder and CheckpointManager appends /checkpoint/)
    output_dir = None
    for d in sorted(dcp_dirs, key=lambda d: os.path.getmtime(d), reverse=True):
        parts = d.split(os.sep)
        for depth in range(len(parts), 0, -1):
            potential = os.sep.join(parts[:depth])
            if os.path.basename(potential).startswith("step-"):
                # step-N -> checkpoint/ -> dump_folder
                ckpt_parent = os.path.dirname(potential)
                output_dir = os.path.dirname(ckpt_parent)
                break
        if output_dir:
            break

    if not output_dir:
        print("No VLM checkpoint found")
        result = {"val_loss": 999.0, "output_dir": None}
        with open(cache_path, "w") as f:
            json.dump(result, f)
        sys.exit(1)

    # Find data and tokenizer
    data_path = None
    for dp in [f"{workspace}/data/cc12m", f"{workspace}/tests/assets/cc12m_test"]:
        if os.path.isdir(dp):
            data_path = dp
            break

    tok_path = None
    for tp in [f"{workspace}/assets/hf/Qwen3-0.6B", f"{workspace}/tests/assets/tokenizer"]:
        if os.path.isdir(tp):
            tok_path = tp
            break

    if not data_path or not tok_path:
        print(f"Missing data or tokenizer (data={data_path}, tok={tok_path})")
        result = {"val_loss": 999.0, "output_dir": output_dir}
        with open(cache_path, "w") as f:
            json.dump(result, f)
        sys.exit(1)

    import subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = src_dir
    cmd = [
        sys.executable, "-m", "torchtitan.experiments.vlm.evaluate",
        "--flavor", "qwen3_0.6B",
        "--checkpoint_dir", output_dir,
        "--data_path", data_path,
        "--tokenizer_path", tok_path,
        "--steps", "10",
        "--batch_size", "4",
    ]
    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=src_dir, timeout=600)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr[-500:], file=sys.stderr)

    metrics_file = os.path.join(output_dir, "eval_metrics.json")
    val_loss = 999.0
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            val_loss = json.load(f).get("val_loss", 999.0)

    result = {"val_loss": val_loss, "output_dir": output_dir}
    with open(cache_path, "w") as f:
        json.dump(result, f)
    print(f"VLM val_loss: {val_loss:.4f}")


# ---------------------------------------------------------------------------
# Flux evaluation
# ---------------------------------------------------------------------------


def eval_flux(args: list[str]) -> None:
    cache_file, workspace = args[0], args[1]
    from pathlib import Path

    ws = Path(workspace)
    diagnostics_files = sorted(ws.glob("**/diagnostics.json"))
    log_files = sorted(ws.glob("**/*.log")) + [p for p in (ws / "nohup.out",) if p.exists()]

    best_cosine = 1.0
    best_mse = 0.0
    best_variance = 0.0
    last_loss = float("inf")
    candidates = []

    for path in diagnostics_files:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        candidate = {
            "path": str(path.parent.relative_to(ws)),
            "cond_uncond_cosine": float(data.get("cond_uncond_cosine", 1.0)),
            "cond_uncond_mse": float(data.get("cond_uncond_mse", 0.0)),
            "output_variance": float(data.get("output_variance", 0.0)),
            "timestep_cosine": float(data.get("timestep_cosine", 1.0)),
        }
        candidates.append(candidate)
        best_cosine = min(best_cosine, candidate["cond_uncond_cosine"])
        best_mse = max(best_mse, candidate["cond_uncond_mse"])
        best_variance = max(best_variance, candidate["output_variance"])

    for path in log_files:
        try:
            for line in path.read_text().splitlines():
                match = re.search(r"global_avg_loss[:\s]+([0-9.]+)", line)
                if match:
                    last_loss = min(last_loss, float(match.group(1)))
                    continue
                match = re.search(r"\bloss[=:\s]+([0-9.]+)", line, re.IGNORECASE)
                if match:
                    last_loss = min(last_loss, float(match.group(1)))
        except Exception:
            continue

    best_timestep_cosine = min(
        (c["timestep_cosine"] for c in candidates), default=1.0
    )
    result = {
        "cond_uncond_cosine": best_cosine,
        "cond_uncond_mse": best_mse,
        "output_variance": best_variance,
        "timestep_cosine": best_timestep_cosine,
        "last_loss": last_loss,
        "candidates": candidates,
    }

    cache_path = ws / cache_file
    cache_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print("Flux metrics:", json.dumps(result, sort_keys=True))
    sys.exit(0 if diagnostics_files else 1)


# ---------------------------------------------------------------------------
# MoE evaluation
# ---------------------------------------------------------------------------


def eval_moe(args: list[str]) -> None:
    from collections import defaultdict
    from pathlib import Path
    from typing import Callable

    task_slug, cache_file, workspace = args[0], args[1], args[2]
    src_dir = os.environ.get("SRC_DIR", "/mcp_server")

    sys.path.insert(0, workspace)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    os.chdir(workspace)

    import math
    import subprocess
    import traceback

    import torch

    Check = tuple[str, Callable[[str], None]]

    def _with_deterministic(check: Callable[[], None]) -> None:
        prev = torch.are_deterministic_algorithms_enabled()
        prev_fill = torch.utils.deterministic.fill_uninitialized_memory
        torch.use_deterministic_algorithms(True)
        try:
            check()
        finally:
            torch.utils.deterministic.fill_uninitialized_memory = prev_fill
            torch.use_deterministic_algorithms(prev)

    def _check_ff_biases(ws: str) -> None:
        from torchtitan.models.common.feed_forward import FeedForward

        def _run():
            m = FeedForward.Config(hidden_dim=16, bias=True).build(dim=8)
            m.init_weights(init_std=0.02)
            for name, bias in [("w1.bias", m.w1.bias), ("w2.bias", m.w2.bias), ("w3.bias", m.w3.bias)]:
                assert bias is not None
                assert torch.isfinite(bias).all(), f"{name} non-finite"
                assert torch.count_nonzero(bias).item() == 0, f"{name} not zero-init"

        _with_deterministic(_run)

    def _check_router_bias(ws: str) -> None:
        from torchtitan.models.common.moe.moe import TokenChoiceTopKRouter

        def _run():
            r = TokenChoiceTopKRouter(dim=8, num_experts=4, num_expert_groups=None, num_limited_groups=None, top_k=2, score_func="softmax", route_norm=False, route_scale=1.0, gate_bias=True)
            r.init_weights(init_std=0.02)
            assert r.gate.bias is not None
            assert torch.isfinite(r.gate.bias).all()
            assert torch.count_nonzero(r.gate.bias).item() == 0

        _with_deterministic(_run)

    def _parse_losses(raw: str) -> dict[int, float]:
        losses = {}
        for line in raw.splitlines():
            line = _ANSI_RE.sub("", line)
            if "validate step:" in line:
                continue
            m = re.search(r"step:\s*(\d+)\s+loss:\s*([^\s]+)", line)
            if m:
                try:
                    losses[int(m.group(1))] = float(m.group(2))
                except ValueError:
                    pass
        return losses

    def _run_seed(ws: str, seed: int, steps: int = 5) -> dict:
        out_dir = os.path.join(ws, "outputs/moe_eval_hidden")
        os.makedirs(out_dir, exist_ok=True)
        cmd = ["bash", os.path.join(ws, "run_train.sh"), "--debug.deterministic", f"--debug.seed={seed}", f"--training.steps={steps}", f"--dump_folder=outputs/moe_eval_hidden/seed_{seed}"]
        env = {**os.environ, "NGPU": "1", "MODULE": "gpt_oss", "CONFIG": "gpt_oss_debugmodel", "PYTHONPATH": f"{ws}:{os.environ.get('PYTHONPATH', '')}"}
        try:
            proc = subprocess.run(cmd, cwd=ws, env=env, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            return {"seed": seed, "timeout": True, "losses": {}}
        return {"seed": seed, "returncode": proc.returncode, "timeout": False, "losses": _parse_losses(proc.stdout + "\n" + proc.stderr)}

    def _check_seed(ws: str, seed: int) -> None:
        r = _run_seed(ws, seed)
        if r["timeout"]:
            raise AssertionError(f"Seed {seed} timed out")
        if r.get("returncode", 1) != 0:
            raise AssertionError(f"Seed {seed} failed (rc={r['returncode']})")
        losses = r["losses"]
        missing = [s for s in range(1, 6) if s not in losses]
        if missing:
            raise AssertionError(f"Seed {seed} missing steps {missing}")
        bad = [(s, losses[s]) for s in range(1, 6) if not math.isfinite(losses[s])]
        if bad:
            raise AssertionError(f"Seed {seed} non-finite losses: {bad}")
        if losses[5] >= losses[1]:
            raise AssertionError(f"Seed {seed} didn't improve: step1={losses[1]:.4f} step5={losses[5]:.4f}")

    def _check_load_balance(ws: str) -> None:
        fp = os.path.join(ws, "torchtitan/models/deepseek_v3/__init__.py")
        if not os.path.exists(fp):
            raise AssertionError(f"File not found: {fp}")
        if "load_balance_coeff=None" in open(fp).read():
            raise AssertionError("load_balance_coeff=None still present")

    def _extract_moe_metrics(ws: str) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for mf in sorted(glob.glob(f"{ws}/**/metrics.json", recursive=True)):
            try:
                with open(mf) as f:
                    m = json.load(f)
                if "loss" in m:
                    metrics["loss"] = float(m["loss"])
                    break
            except Exception:
                continue
        if "loss" not in metrics:
            for lf in sorted(glob.glob(f"{ws}/**/*.log", recursive=True)):
                try:
                    for line in open(lf).read().splitlines():
                        line = _ANSI_RE.sub("", line)
                        m = re.search(r"step:\s*\d+\s+loss:\s*([0-9.]+)", line)
                        if m:
                            metrics["loss"] = float(m.group(1))
                except Exception:
                    continue
        for ef in sorted(glob.glob(f"{ws}/**/expert_stats.json", recursive=True)):
            try:
                with open(ef) as f:
                    stats = json.load(f)
                tokens = stats.get("tokens_per_expert", [])
                if tokens and sum(tokens) > 0:
                    n = len(tokens)
                    metrics["expert_balance"] = n * min(tokens) / sum(tokens)
                    break
            except Exception:
                continue
        return metrics

    CHECKS: dict[str, list[Check]] = {
        "moe_seed_sweep": [
            ("feed_forward_biases_initialized", _check_ff_biases),
            ("router_bias_initialized", _check_router_bias),
            *[(f"gpt_oss_seed_{s}_converges", lambda ws, s=s: _check_seed(ws, s)) for s in (0, 42, 123, 999)],
        ],
        "moe_debug_balance": [
            ("load_balance_fixed", _check_load_balance),
        ],
    }
    METRIC_EXTRACTORS = {"moe_debug_balance": _extract_moe_metrics}

    if task_slug not in CHECKS:
        print(f"Unknown moe task: {task_slug}. Available: {sorted(CHECKS)}", file=sys.stderr)
        sys.exit(1)

    results = []
    passed = 0
    for name, check in CHECKS[task_slug]:
        try:
            check(workspace)
            passed += 1
            results.append({"name": name, "status": "passed"})
        except Exception as exc:
            results.append({"name": name, "status": "failed", "error": str(exc), "traceback": traceback.format_exc(limit=8)})

    total = len(results)
    payload = {"task_slug": task_slug, "pass_rate": round(passed / total, 4) if total else 0.0, "checks_passed": passed, "checks_total": total, "checks": results}

    extractor = METRIC_EXTRACTORS.get(task_slug)
    if extractor:
        payload.update(extractor(workspace))

    cache_path = os.path.join(workspace, cache_file)
    with open(cache_path, "w") as f:
        json.dump(payload, indent=2, sort_keys=True, fp=f)
    print(json.dumps(payload, indent=2, sort_keys=True))
    sys.exit(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

EVAL_TYPES = {
    "emb": eval_emb,
    "vlm": eval_vlm,
    "flux": eval_flux,
    "moe": eval_moe,
}

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

eval_type = sys.argv[1]
if eval_type not in EVAL_TYPES:
    print(f"Unknown eval type: {eval_type}. Available: {sorted(EVAL_TYPES)}")
    sys.exit(1)

EVAL_TYPES[eval_type](sys.argv[2:])
