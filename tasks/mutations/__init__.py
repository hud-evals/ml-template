"""Mutation helpers for audit scenarios."""

from .data import (
    VALID_DATA_MUTATIONS,
    contaminate,
    fingerprint_pair,
    inject_data_leakage,
    inject_duplicates,
    inject_label_noise,
)
from .eval import (
    VALID_EVAL_MUTATIONS,
    contaminate_eval_signal,
    inject_eval_leakage,
)

__all__ = [
    "VALID_DATA_MUTATIONS",
    "VALID_EVAL_MUTATIONS",
    "contaminate",
    "contaminate_eval_signal",
    "fingerprint_pair",
    "inject_data_leakage",
    "inject_duplicates",
    "inject_eval_leakage",
    "inject_label_noise",
]
