"""Checks that stop bad data or leaky splits from reaching a model."""

from collections.abc import Iterable

import pandas as pd


def validate_required_columns(df: pd.DataFrame, columns: Iterable[str]) -> bool:
    """Return True if every column exists, otherwise raise ValueError."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def validate_binary_labels(labels: pd.Series) -> bool:
    """Return True if labels contain only 0 and 1, otherwise raise ValueError."""
    unexpected = set(labels.unique()) - {0, 1}
    if unexpected:
        raise ValueError(f"Labels must be 0 or 1, found: {sorted(unexpected)}")
    return True


def validate_no_text_overlap(*splits: pd.DataFrame) -> bool:
    """Return True if no text appears in two splits, otherwise raise ValueError."""
    seen: set[str] = set()
    for position, split in enumerate(splits):
        texts = set(split["text"])
        shared = seen & texts
        if shared:
            raise ValueError(
                f"Data leakage: split {position} shares {len(shared)} texts "
                "with an earlier split"
            )
        seen |= texts
    return True
