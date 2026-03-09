"""Grading check scripts for training scenarios.

Each function returns a Python script string that can be written to /tmp and executed.
The grader runs these to verify agent outputs.
"""


def grader_eval_script(src_dir: str) -> str:
    """Script to find checkpoints, evaluate on MTEB SciFact, cache best result."""
    return f'''
import glob, json, os, sys

workspace = sys.argv[1]
src_dir = "{src_dir}"
cache_path = "/tmp/grader_eval.json"
MAX_EVALS = 10
GOLDEN_THRESHOLD = 0.65

ckpt_dirs = set()
for pattern in ["**/model.safetensors", "**/pytorch_model.bin"]:
    for wf in glob.glob(f"{{workspace}}/{{pattern}}", recursive=True):
        ckpt_dirs.add(os.path.dirname(wf))

if not ckpt_dirs:
    print("No model checkpoints found")
    sys.exit(1)

ckpt_dirs = sorted(ckpt_dirs, key=lambda d: os.path.getmtime(d), reverse=True)

sys.path.insert(0, src_dir)
from torchtitan.experiments.embedding.evaluate import evaluate_mteb

best_ndcg = -1
best_dir = None
for idx, d in enumerate(ckpt_dirs[:MAX_EVALS]):
    try:
        metrics = evaluate_mteb(d, ["SciFact"])
        ndcg = metrics.get("ndcg@10", 0)
        print(f"[{{idx+1}}/{{min(len(ckpt_dirs), MAX_EVALS)}}] {{d}}: nDCG@10={{ndcg:.4f}}")
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_dir = d
        if best_ndcg >= GOLDEN_THRESHOLD:
            print(f"Above golden threshold ({{GOLDEN_THRESHOLD}}), stopping early")
            break
    except Exception as e:
        print(f"{{d}}: eval failed ({{e}})")

result = {{"ndcg@10": best_ndcg, "best_dir": best_dir}}
with open(cache_path, "w") as f:
    json.dump(result, f)
print(f"Best: {{best_dir}} nDCG@10={{best_ndcg:.4f}}")
'''


def ndcg_check_script() -> str:
    """Script to check nDCG@10 against a threshold."""
    return '''
import json, sys

threshold = float(sys.argv[1])
cache_path = "/tmp/grader_eval.json"

with open(cache_path) as f:
    result = json.load(f)
ndcg = result.get("ndcg@10", 0)
print(f"nDCG@10: {ndcg:.4f}, threshold: {threshold}")
sys.exit(0 if ndcg >= threshold else 1)
'''


def metadata_check_script() -> str:
    """Script to check for training_metadata.json with a given stage."""
    return '''
import json, glob, sys, os

stage = sys.argv[1]
workspace = sys.argv[2]

for mf in glob.glob(f"{workspace}/**/training_metadata.json", recursive=True):
    with open(mf) as f:
        m = json.load(f)
    if m.get("stage") == stage:
        print(os.path.dirname(mf))
        sys.exit(0)

sys.exit(1)
'''


def resume_check_script() -> str:
    """Script to check for a finetune checkpoint that resumed from a prior checkpoint."""
    return '''
import json, glob, sys, os

workspace = sys.argv[1]

for mf in glob.glob(f"{workspace}/**/training_metadata.json", recursive=True):
    with open(mf) as f:
        m = json.load(f)
    if m.get("stage") == "finetune" and m.get("resume_from"):
        print(os.path.dirname(mf))
        sys.exit(0)

sys.exit(1)
'''


def merge_check_script() -> str:
    """Script to check for merge_metadata.json with 2+ source checkpoints."""
    return '''
import json, glob, sys, os

workspace = sys.argv[1]

for mf in glob.glob(f"{workspace}/**/merge_metadata.json", recursive=True):
    with open(mf) as f:
        m = json.load(f)
    if len(m.get("source_checkpoints", [])) >= 2:
        print(os.path.dirname(mf))
        sys.exit(0)

sys.exit(1)
'''


# ---------------------------------------------------------------------------
# VLM grading scripts
# ---------------------------------------------------------------------------


def vlm_metadata_check_script() -> str:
    """Script to check for VLM training_metadata.json."""
    return '''
import json, glob, sys, os

workspace = sys.argv[1]

for mf in glob.glob(f"{workspace}/**/training_metadata.json", recursive=True):
    with open(mf) as f:
        m = json.load(f)
    if m.get("task", "").startswith("vlm"):
        print(os.path.dirname(mf))
        sys.exit(0)

sys.exit(1)
'''


def vlm_checkpoint_check_script() -> str:
    """Script to check for VLM DCP checkpoint files."""
    return '''
import glob, sys, os

workspace = sys.argv[1]

# DCP checkpoints have .distcp files or __0_0.distcp etc
for pattern in ["**/checkpoint/**/*.distcp", "**/checkpoint/**/.__0_0.distcp"]:
    files = glob.glob(f"{workspace}/{pattern}", recursive=True)
    if files:
        ckpt_dir = os.path.dirname(files[0])
        print(f"Found DCP checkpoint: {ckpt_dir}")
        sys.exit(0)

# Also check for safetensors (if agent converted to HF format)
for pattern in ["**/model.safetensors", "**/pytorch_model.bin"]:
    files = glob.glob(f"{workspace}/{pattern}", recursive=True)
    if files:
        print(f"Found model weights: {files[0]}")
        sys.exit(0)

print("No checkpoints found")
sys.exit(1)
'''


def vlm_eval_script(src_dir: str) -> str:
    """Script to evaluate VLM checkpoint by computing validation loss."""
    return f'''
import json, glob, os, sys, subprocess

workspace = sys.argv[1]
src_dir = "{src_dir}"
cache_path = "/tmp/vlm_eval.json"

# Find training output directory (has training_metadata.json)
output_dir = None
for mf in glob.glob(f"{{workspace}}/**/training_metadata.json", recursive=True):
    with open(mf) as f:
        m = json.load(f)
    if m.get("task", "").startswith("vlm"):
        output_dir = os.path.dirname(mf)
        break

if not output_dir:
    print("No VLM training metadata found")
    result = {{"val_loss": 999.0, "output_dir": None}}
    with open(cache_path, "w") as f:
        json.dump(result, f)
    sys.exit(1)

# Find data and tokenizer paths
data_path = None
for dp in [f"{{workspace}}/data/cc12m", f"{{workspace}}/tests/assets/cc12m_test"]:
    if os.path.isdir(dp):
        data_path = dp
        break
tok_path = None
for tp in [f"{{workspace}}/tokenizer", f"{{workspace}}/tests/assets/tokenizer"]:
    if os.path.isdir(tp):
        tok_path = tp
        break

if not data_path or not tok_path:
    print(f"Missing data or tokenizer (data={{data_path}}, tok={{tok_path}})")
    result = {{"val_loss": 999.0, "output_dir": output_dir}}
    with open(cache_path, "w") as f:
        json.dump(result, f)
    sys.exit(1)

# Run evaluate
env = os.environ.copy()
env["PYTHONPATH"] = workspace
cmd = [
    sys.executable, "-m", "torchtitan.experiments.vlm.evaluate",
    "--checkpoint_dir", output_dir,
    "--data_path", data_path,
    "--tokenizer_path", tok_path,
    "--steps", "10",
    "--batch_size", "4",
]
print(f"Running: {{' '.join(cmd)}}")
proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
print(proc.stdout)
if proc.stderr:
    print(proc.stderr[-500:], file=sys.stderr)

# Read eval metrics
metrics_file = os.path.join(output_dir, "eval_metrics.json")
if os.path.exists(metrics_file):
    with open(metrics_file) as f:
        metrics = json.load(f)
    val_loss = metrics.get("val_loss", 999.0)
else:
    val_loss = 999.0

result = {{"val_loss": val_loss, "output_dir": output_dir}}
with open(cache_path, "w") as f:
    json.dump(result, f)
print(f"VLM val_loss: {{val_loss:.4f}}")
sys.exit(0)
'''


def vlm_loss_check_script() -> str:
    """Script to check VLM validation loss against a threshold."""
    return '''
import json, sys

threshold = float(sys.argv[1])
cache_path = "/tmp/vlm_eval.json"

with open(cache_path) as f:
    result = json.load(f)
val_loss = result.get("val_loss", 999.0)
print(f"val_loss: {val_loss:.4f}, threshold: {threshold}")
sys.exit(0 if val_loss <= threshold else 1)
'''
