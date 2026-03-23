"""Require merged checkpoints to beat non-merged checkpoints."""
import json, os, sys

threshold = float(sys.argv[1])
workspace = sys.argv[2]
cache_path = os.path.join(workspace, ".emb_eval.json")

with open(cache_path) as f:
    result = json.load(f)

candidates = result.get("candidates", [])
merged = [c.get("ndcg@10", 0.0) for c in candidates if c.get("is_merged")]
sources = [c.get("ndcg@10", 0.0) for c in candidates if not c.get("is_merged")]
if not merged:
    print("No merged checkpoints were evaluated")
    sys.exit(1)
if not sources:
    print("No source checkpoints were evaluated")
    sys.exit(1)

delta = max(merged) - max(sources)
print(f"best_merged-max_source delta: {delta:.4f}, threshold: {threshold:.4f}")
sys.exit(0 if delta >= threshold else 1)
