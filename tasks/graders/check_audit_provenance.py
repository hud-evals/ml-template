"""Ensure the agent modified the contaminated training data before training.

Checks that the training data file (data/scifact.jsonl by default) has been
modified from its original contaminated state. Works by comparing the current
file hash against the hash stored during contamination setup.
"""
import hashlib
import json
import os
import sys

workspace = sys.argv[1]
info_path = "/tmp/.grader_contamination_info.json"
if not os.path.exists(info_path):
    print("No contamination info found (was setup run?)")
    sys.exit(1)

with open(info_path) as f:
    info = json.load(f)

# Check that the training data file was modified from its contaminated state
train_file = info.get("train_file", "data/scifact.jsonl")
train_path = os.path.join(workspace, train_file)

if not os.path.exists(train_path):
    print(f"Training data file not found: {train_path}")
    sys.exit(1)

current_hash = hashlib.md5(open(train_path, "rb").read()).hexdigest()
if current_hash != info["hash"]:
    print(f"Training data was modified (provenance verified): {train_path}")
    sys.exit(0)

print(f"Training data unchanged from contaminated version: {train_path}")
sys.exit(1)
