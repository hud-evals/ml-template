"""Check a cached metric against a threshold.

Usage: check_threshold.py <cache_file> <metric_key> <threshold> <workspace> [lower]

Reads *metric_key* from *workspace*/<cache_file> and compares to *threshold*.
Default comparison is >= (higher is better).  Pass "lower" as the 5th arg
for <= (lower is better).
"""
import json, os, sys

cache_file = sys.argv[1]
metric_key = sys.argv[2]
threshold = float(sys.argv[3])
workspace = sys.argv[4]
lower_is_better = len(sys.argv) > 5 and sys.argv[5] == "lower"

cache_path = os.path.join(workspace, cache_file)
with open(cache_path) as f:
    result = json.load(f)

default = float("inf") if lower_is_better else 0.0
value = float(result.get(metric_key, default))
passed = value <= threshold if lower_is_better else value >= threshold
print(f"{metric_key}: {value:.4f}, threshold: {threshold}")
sys.exit(0 if passed else 1)
