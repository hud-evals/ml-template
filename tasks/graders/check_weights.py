"""Verify model checkpoint files exist in the workspace.

Usage:
  check_weights.py <workspace>

Searches for DCP checkpoints (step-N/.metadata) and HF checkpoints
(model.safetensors). Exits 0 if at least one is found.
"""
import glob
import os
import sys

workspace = sys.argv[1]

_EXCLUDE = ("assets/", "tests/", ".venv/")

found = []

# DCP checkpoints
for mf in glob.glob(f"{workspace}/**/.metadata", recursive=True):
    d = os.path.dirname(mf)
    rel = os.path.relpath(d, workspace)
    if any(rel.startswith(p) for p in _EXCLUDE):
        continue
    if os.path.basename(d).startswith("step-"):
        found.append(d)

# HF checkpoints (from merge or manual export)
for pattern in ["**/model.safetensors", "**/model-*.safetensors", "**/model.safetensors.index.json", "**/pytorch_model.bin"]:
    for wf in glob.glob(f"{workspace}/{pattern}", recursive=True):
        rel = os.path.relpath(wf, workspace)
        if any(rel.startswith(p) for p in _EXCLUDE):
            continue
        found.append(os.path.dirname(wf))

if found:
    for d in sorted(set(found)):
        print(f"  {d}")
    print(f"Found {len(set(found))} checkpoint(s)")
    sys.exit(0)

print("No checkpoints found")
sys.exit(1)
