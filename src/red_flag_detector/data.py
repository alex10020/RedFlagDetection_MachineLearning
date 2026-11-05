"""Load every dataset into one shape: a `text` column and a binary `label`."""

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .validation import validate_binary_labels, validate_required_columns

JIGSAW_TYPES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
STRESS_COLUMNS = ["id", "text", "label", "category", "topic", "label_status"]
GASLIGHTING_COLUMNS = ["id", "turn", "speaker role", "utterance", "category"]


def clean_text(text: str) -> str:
    """Collapse whitespace. Case and punctuation stay because they carry tone."""
    return re.sub(r"\s+", " ", str(text)).strip()


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. See data/README.md")
    return pd.read_csv(path)


def load_jigsaw(path: str | Path) -> pd.DataFrame:
    """Load D2 (Jigsaw train.csv). A comment counts as toxic if any type is set."""
    raw = _read_csv(path)
    validate_required_columns(raw, ["comment_text", *JIGSAW_TYPES])
    data = raw[JIGSAW_TYPES].copy()
    data.insert(0, "text", raw["comment_text"].map(clean_text))
    data.insert(1, "label", raw[JIGSAW_TYPES].max(axis=1).astype(int))
    validate_binary_labels(data["label"])
    return data


def load_toxic_conversations(split: str = "train", sample: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Load D1 from Hugging Face, with an optional stratified sample."""
    from datasets import load_dataset

    raw = load_dataset("SetFit/toxic_conversations", split=split).to_pandas()
    validate_required_columns(raw, ["text", "label"])
    data = pd.DataFrame({"text": raw["text"].map(clean_text), "label": raw["label"].astype(int)})
    validate_binary_labels(data["label"])
    if sample is not None and sample < len(data):
        data, _ = train_test_split(data, train_size=sample, stratify=data["label"], random_state=seed)
    return data.reset_index(drop=True)


def load_gaslighting(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load D3 assistant messages, labelled 1 for gaslighting and 0 for safe.

    Pass the JSON files of one official split, both classes together. We keep
    assistant turns only, because the label describes the assistant's behaviour
    and user turns come from the same scenarios in both classes. We also drop
    `internal thought`: it states the speaker's intent, so it would leak the label.
    Keep the official splits, because safe and gaslighting chats share scenarios.
    """
    frames = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}. See data/README.md")
        frames.append(pd.read_json(path))
    raw = pd.concat(frames, ignore_index=True)
    validate_required_columns(raw, GASLIGHTING_COLUMNS)
    unexpected = set(raw["category"].unique()) - {"gaslighting", "safe"}
    if unexpected:
        raise ValueError(f"Unexpected D3 categories: {sorted(unexpected)}")
    assistant = raw[raw["speaker role"] == "assistant"]
    return pd.DataFrame(
        {
            "text": assistant["utterance"].map(clean_text),
            "label": assistant["category"].eq("gaslighting").astype(int),
            "chat_id": assistant["id"],
        }
    ).reset_index(drop=True)


def load_stress_test(path: str | Path) -> pd.DataFrame:
    """Load our own dating-style test set."""
    data = _read_csv(path)
    validate_required_columns(data, STRESS_COLUMNS)
    validate_binary_labels(data["label"])
    data["text"] = data["text"].map(clean_text)
    return data


def deduplicate(data: pd.DataFrame) -> pd.DataFrame:
    """Drop repeated texts, ignoring case, so copies cannot cross splits.

    A text that appears with both labels goes too, because we cannot say which
    label is right.
    """
    keys = data["text"].str.lower()
    conflicting = data.groupby(keys)["label"].transform("nunique") > 1
    kept = data[~conflicting].copy()
    return kept[~kept["text"].str.lower().duplicated()].reset_index(drop=True)


def make_splits(
    data: pd.DataFrame,
    val_size: float = 0.2,
    test_size: float | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, ...]:
    """Split with the same toxic share in each part.

    Pass `test_size` only for a dataset without a labelled official test set.
    Sizes are shares of the full data.
    """
    if test_size:
        rest, test = train_test_split(data, test_size=test_size, stratify=data["label"], random_state=seed)
        train, val = train_test_split(
            rest, test_size=val_size / (1 - test_size), stratify=rest["label"], random_state=seed
        )
        return train, val, test
    return tuple(train_test_split(data, test_size=val_size, stratify=data["label"], random_state=seed))
