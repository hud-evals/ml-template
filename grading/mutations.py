"""Data mutation functions for adversarial scenarios.

These functions inject specific types of data contamination into training data,
creating scenarios where the agent must identify and fix data quality issues.
"""

import json
import random


def inject_label_noise(
    data_path: str,
    output_path: str,
    noise_rate: float = 0.3,
    seed: int = 42,
) -> dict:
    """Swap positive and a random negative for a fraction of training pairs.

    This creates pairs where the "positive" is actually a hard negative and vice versa,
    degrading training quality. The agent must detect the noise pattern and clean the data.

    Returns metadata about the mutation for grading.
    """
    random.seed(seed)

    pairs = []
    with open(data_path) as f:
        for line in f:
            pairs.append(json.loads(line))

    n_corrupted = 0
    corrupted_indices = []
    for i, pair in enumerate(pairs):
        if random.random() < noise_rate and pair.get("negatives"):
            neg_idx = random.randint(0, len(pair["negatives"]) - 1)
            pair["positive"], pair["negatives"][neg_idx] = pair["negatives"][neg_idx], pair["positive"]
            n_corrupted += 1
            corrupted_indices.append(i)

    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    return {
        "mutation": "label_noise",
        "noise_rate": noise_rate,
        "total_pairs": len(pairs),
        "corrupted_pairs": n_corrupted,
        "corrupted_indices": corrupted_indices,
    }


def inject_data_leakage(
    train_path: str,
    test_path: str,
    output_path: str,
    leak_rate: float = 0.2,
    seed: int = 42,
) -> dict:
    """Copy a fraction of test data into the training set.

    This creates train/test contamination. The model will memorize test examples,
    giving artificially high eval scores but poor generalization.
    The agent must detect the leakage and remove duplicates.

    Returns metadata about the mutation for grading.
    """
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
    }


def inject_duplicates(
    data_path: str,
    output_path: str,
    dup_rate: float = 0.3,
    seed: int = 42,
) -> dict:
    """Duplicate a subset of training pairs to bias the model.

    Over-represented examples dominate gradient updates, causing the model to
    overfit to specific patterns. The agent must detect and deduplicate.

    Returns metadata about the mutation for grading.
    """
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
    }
