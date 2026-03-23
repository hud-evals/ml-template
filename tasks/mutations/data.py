"""Training-data mutation functions for audit scenarios."""

from __future__ import annotations

import hashlib
import json
import os
import random
from copy import deepcopy

VALID_DATA_MUTATIONS = {"label_noise", "data_leakage", "duplicates"}


def fingerprint_pair(pair: dict) -> str:
    payload = json.dumps(pair, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()


def inject_label_noise(
    data_path: str,
    output_path: str,
    noise_rate: float = 0.3,
    seed: int = 42,
) -> dict:
    """Swap positive and a random negative for a fraction of training pairs."""
    random.seed(seed)

    pairs = []
    with open(data_path) as f:
        for line in f:
            pairs.append(json.loads(line))

    n_corrupted = 0
    corrupted_pairs = []
    for i, pair in enumerate(pairs):
        if random.random() < noise_rate and pair.get("negatives"):
            before = deepcopy(pair)
            neg_idx = random.randint(0, len(pair["negatives"]) - 1)
            pair["positive"], pair["negatives"][neg_idx] = pair["negatives"][neg_idx], pair["positive"]
            n_corrupted += 1
            corrupted_pairs.append({
                "index": i,
                "query": before.get("query", ""),
                "clean_fingerprint": fingerprint_pair(before),
                "mutated_fingerprint": fingerprint_pair(pair),
                "clean_positive": before.get("positive", ""),
                "mutated_positive": pair.get("positive", ""),
            })

    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    return {
        "mutation": "label_noise",
        "noise_rate": noise_rate,
        "total_pairs": len(pairs),
        "corrupted_pairs": n_corrupted,
        "corrupted_examples": corrupted_pairs,
    }


def inject_data_leakage(
    train_path: str,
    test_path: str,
    output_path: str,
    leak_rate: float = 0.2,
    seed: int = 42,
) -> dict:
    """Copy a fraction of test data into the training set."""
    random.seed(seed)

    train_pairs = []
    with open(train_path) as f:
        for line in f:
            train_pairs.append(json.loads(line))

    test_pairs = []
    with open(test_path) as f:
        for line in f:
            test_pairs.append(json.loads(line))

    n_leak = max(1, int(len(test_pairs) * leak_rate))
    leaked = random.sample(test_pairs, min(n_leak, len(test_pairs)))

    combined = train_pairs + leaked
    random.shuffle(combined)

    with open(output_path, "w") as f:
        for pair in combined:
            f.write(json.dumps(pair) + "\n")

    return {
        "mutation": "data_leakage",
        "leak_rate": leak_rate,
        "original_train": len(train_pairs),
        "leaked_test": len(leaked),
        "total_output": len(combined),
        "leaked_fingerprints": [fingerprint_pair(pair) for pair in leaked],
    }


def inject_duplicates(
    data_path: str,
    output_path: str,
    dup_rate: float = 0.3,
    seed: int = 42,
) -> dict:
    """Duplicate a subset of training pairs to bias the model."""
    random.seed(seed)

    pairs = []
    with open(data_path) as f:
        for line in f:
            pairs.append(json.loads(line))

    n_to_dup = max(1, int(len(pairs) * dup_rate))
    dup_indices = random.sample(range(len(pairs)), min(n_to_dup, len(pairs)))
    duplicates = [pairs[i] for i in dup_indices]

    n_copies = random.randint(3, 8)
    for _ in range(n_copies):
        pairs.extend(duplicates)

    random.shuffle(pairs)

    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    return {
        "mutation": "duplicates",
        "dup_rate": dup_rate,
        "original_count": len(pairs) - len(duplicates) * n_copies,
        "duplicated_pairs": len(dup_indices),
        "copies": n_copies,
        "total_output": len(pairs),
        "duplicated_fingerprints": [fingerprint_pair(pair) for pair in duplicates],
    }


def contaminate(
    workspace: str,
    mutation: str,
    *,
    train_file: str = "data/scifact.jsonl",
    val_file: str = "data/val.jsonl",
    noise_rate: float = 0.3,
    leak_rate: float = 0.2,
) -> dict:
    """Apply a training-data mutation and write metadata to /tmp."""
    if mutation not in VALID_DATA_MUTATIONS:
        raise ValueError(f"Unknown mutation: {mutation}")

    train_path = f"{workspace}/{train_file}"

    if mutation == "label_noise":
        info = inject_label_noise(train_path, train_path, noise_rate=noise_rate)
    elif mutation == "data_leakage":
        val_path = f"{workspace}/{val_file}"
        info = inject_data_leakage(train_path, val_path, train_path, leak_rate=leak_rate)
    else:
        info = inject_duplicates(train_path, train_path, dup_rate=noise_rate)

    contamination_info = {
        "type": mutation,
        "hash": hashlib.md5(open(train_path, "rb").read()).hexdigest(),
        "line_count": sum(1 for _ in open(train_path)),
        "train_file": train_file,
        **info,
    }
    info_path = "/tmp/.grader_contamination_info.json"
    with open(info_path, "w") as f:
        json.dump(contamination_info, f)
    os.chmod(info_path, 0o600)
    return contamination_info
