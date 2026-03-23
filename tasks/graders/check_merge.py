"""Check for merge_metadata.json with 2+ source checkpoints."""
import json, glob, sys, os

workspace = sys.argv[1]

for mf in glob.glob(f"{workspace}/**/merge_metadata.json", recursive=True):
    with open(mf) as f:
        m = json.load(f)
    if len(m.get("source_checkpoints", [])) >= 2:
        print(os.path.dirname(mf))
        sys.exit(0)

sys.exit(1)
