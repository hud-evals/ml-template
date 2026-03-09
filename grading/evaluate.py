"""Grader-side evaluation utilities.

Thin wrappers around the embedding experiment's evaluation code,
used by the grading scripts to evaluate agent checkpoints.
"""

import sys


def evaluate_checkpoint(model_path: str, tasks: list[str], src_dir: str) -> dict:
    """Evaluate a checkpoint on MTEB tasks. Returns metrics dict."""
    sys.path.insert(0, src_dir)
    from torchtitan.experiments.embedding.evaluate import evaluate_mteb
    return evaluate_mteb(model_path, tasks)
