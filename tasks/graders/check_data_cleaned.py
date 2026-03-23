"""Verify the agent modified training data to remove contamination."""
import hashlib, json, sys, os
from collections import Counter

workspace = sys.argv[1]
info_path = "/tmp/.grader_contamination_info.json"

if not os.path.exists(info_path):
    print("No contamination info found (was setup run?)")
    sys.exit(1)

with open(info_path) as f:
    info = json.load(f)

data_path = os.path.join(workspace, "data", "scifact.jsonl")
if not os.path.exists(data_path):
    print("Training data file not found at expected path")
    sys.exit(1)

def fingerprint_pair(pair):
    payload = json.dumps(pair, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()

rows = []
with open(data_path) as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))
counts = Counter(fingerprint_pair(row) for row in rows)

contamination_type = info["type"]
if contamination_type == "label_noise":
    bad = {item["mutated_fingerprint"] for item in info.get("corrupted_examples", [])}
    clean = {item["clean_fingerprint"] for item in info.get("corrupted_examples", [])}
    remaining_bad = sorted(fp for fp in bad if counts.get(fp, 0) > 0)
    recovered = sum(1 for fp in clean if counts.get(fp, 0) > 0)
    if remaining_bad:
        print(f"Mutated label-noise examples still present: {len(remaining_bad)}")
        sys.exit(1)
    if recovered == 0:
        print("No clean replacements for corrupted examples were found")
        sys.exit(1)
    print(f"Recovered {recovered} clean examples and removed all corrupted fingerprints")
    sys.exit(0)

if contamination_type == "data_leakage":
    leaked = info.get("leaked_fingerprints", [])
    remaining = [fp for fp in leaked if counts.get(fp, 0) > 0]
    print(f"Remaining leaked examples: {len(remaining)} / {len(leaked)}")
    sys.exit(0 if not remaining else 1)

if contamination_type == "duplicates":
    duplicated = info.get("duplicated_fingerprints", [])
    offenders = [fp for fp in duplicated if counts.get(fp, 0) > 1]
    print(f"Remaining duplicate offenders: {len(offenders)} / {len(duplicated)}")
    sys.exit(0 if not offenders else 1)

print(f"Unsupported contamination type: {contamination_type}")
sys.exit(1)
