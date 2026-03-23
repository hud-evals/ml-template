"""Verify the visible evaluation file was cleaned after eval-signal contamination."""

import hashlib
import json
import os
import sys
from collections import Counter

workspace = sys.argv[1]
info_path = "/tmp/.grader_eval_signal_info.json"

if not os.path.exists(info_path):
    print("No eval-signal info found (was setup run?)")
    sys.exit(1)

with open(info_path) as f:
    info = json.load(f)

eval_file = info.get("eval_file", "data/val.jsonl")
eval_path = os.path.join(workspace, eval_file)
if not os.path.exists(eval_path):
    print(f"Eval data file not found at expected path: {eval_file}")
    sys.exit(1)


def fingerprint_pair(pair: dict) -> str:
    payload = json.dumps(pair, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()


rows = []
with open(eval_path) as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))
counts = Counter(fingerprint_pair(row) for row in rows)

eval_mutation = info["type"]
if eval_mutation == "eval_leakage":
    leaked = info.get("leaked_fingerprints", [])
    remaining = [fp for fp in leaked if counts.get(fp, 0) > 0]
    print(f"Remaining leaked eval rows: {len(remaining)} / {len(leaked)}")
    sys.exit(0 if not remaining else 1)

print(f"Unsupported eval mutation type: {eval_mutation}")
sys.exit(1)
