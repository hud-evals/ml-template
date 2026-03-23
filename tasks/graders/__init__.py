"""Grader helpers for task definitions.

Each grader is a standalone .py script in this directory. The ``grader()``
helper reads it and returns a grader dict that ``env._grade()`` can execute.
"""
from pathlib import Path
from typing import Any

_DIR = Path(__file__).parent


def grader(script: str, *, args: str = "", name: str | None = None, **kw: Any) -> dict:
    """Build a script-based grader dict from a check file in this directory.

    *script*: filename stem (e.g. ``"check_metadata"``).
    *name*: scoring label; defaults to *script*.
    """
    return {
        "name": name or script,
        "script": (_DIR / f"{script}.py").read_text(),
        "_script_stem": script,
        "args": args,
        **kw,
    }
