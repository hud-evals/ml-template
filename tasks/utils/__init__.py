"""Shared utilities for task definitions."""

from pathlib import Path


def load_patches(task_file: str) -> list[str]:
    """Read all .patch files from the directory containing *task_file*."""
    return [p.read_text() for p in sorted(Path(task_file).parent.glob("*.patch"))]
