"""Check for model checkpoint files in the workspace.

Usage: check_checkpoint.py <workspace> [min_count] [no-merged]

Searches for DCP (.metadata), HF safetensors (including sharded), and
pytorch_model.bin files. Exits 0 if at least *min_count* (default 1)
checkpoints are found.
Pass "no-merged" to exclude directories with "merge" in the path.
"""
import glob, os, sys

workspace = sys.argv[1]
min_count = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "no-merged" else 1
no_merged = "no-merged" in sys.argv[2:]

_EXCLUDE = ("assets/", "tests/", ".venv/")

ckpt_dirs: set[str] = set()

# DCP checkpoints
for mf in glob.glob(f"{workspace}/**/.metadata", recursive=True):
    d = os.path.dirname(mf)
    rel = os.path.relpath(d, workspace)
    if any(rel.startswith(p) for p in _EXCLUDE):
        continue
    ckpt_dirs.add(d)

# HF checkpoints (single file, sharded, or index)
for pattern in ["**/model.safetensors", "**/model-*.safetensors", "**/model.safetensors.index.json", "**/pytorch_model.bin"]:
    for f in glob.glob(f"{workspace}/{pattern}", recursive=True):
        rel = os.path.relpath(f, workspace)
        if any(rel.startswith(p) for p in _EXCLUDE):
            continue
        ckpt_dirs.add(os.path.dirname(f))

# DCP .distcp files
for pattern in ["**/checkpoint/**/*.distcp", "**/checkpoint/**/.__0_0.distcp"]:
    for f in glob.glob(f"{workspace}/{pattern}", recursive=True):
        rel = os.path.relpath(f, workspace)
        if any(rel.startswith(p) for p in _EXCLUDE):
            continue
        ckpt_dirs.add(os.path.dirname(f))

if no_merged:
    ckpt_dirs = {d for d in ckpt_dirs if "merge" not in d.lower()}

print(f"Found {len(ckpt_dirs)} checkpoint(s)")
for d in sorted(ckpt_dirs):
    print(f"  {d}")
sys.exit(0 if len(ckpt_dirs) >= min_count else 1)
