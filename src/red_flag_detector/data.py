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
# Shared D1 sample sizes: the CNN and DistilBERT train on the same 100,000 messages.
D1_TRAIN_SIZE = 100_000
D1_VAL_SIZE = 20_000
# Dating messages S001-S120 are the development set; S121 onwards are the final, unseen test.
DEV_LAST_ID = 120


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


def load_jigsaw_test(test_path: str | Path, labels_path: str | Path) -> pd.DataFrame:
    """Load D2's official test set: Kaggle's test.csv joined with test_labels.csv.

    Kaggle marks the rows it never scored with -1 in every type column; we drop
    them, which leaves the 63,978 labelled comments.
    """
    texts = _read_csv(test_path)
    labels = _read_csv(labels_path)
    validate_required_columns(texts, ["id", "comment_text"])
    validate_required_columns(labels, ["id", *JIGSAW_TYPES])
    raw = texts.merge(labels, on="id", how="inner", validate="one_to_one")
    raw = raw[(raw[JIGSAW_TYPES] != -1).all(axis=1)]
    data = raw[JIGSAW_TYPES].copy()
    data.insert(0, "text", raw["comment_text"].map(clean_text))
    data.insert(1, "label", raw[JIGSAW_TYPES].max(axis=1).astype(int))
    validate_binary_labels(data["label"])
    return data.reset_index(drop=True)


def read_toxic_conversations(path: str | Path, sample: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Read one D1 JSON lines file, with an optional stratified sample."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. See data/README.md")
    raw = pd.read_json(path, lines=True)
    validate_required_columns(raw, ["text", "label"])
    data = pd.DataFrame({"text": raw["text"].map(clean_text), "label": raw["label"].astype(int)})
    validate_binary_labels(data["label"])
    if sample is not None and sample < len(data):
        data, _ = train_test_split(data, train_size=sample, stratify=data["label"], random_state=seed)
    return data.reset_index(drop=True)


def load_toxic_conversations(split: str = "train", sample: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Download D1 from Hugging Face (cached after the first time) and read it.

    We fetch the raw JSON lines file instead of using the `datasets` library,
    because `datasets` needs pyarrow, which some locked-down Windows machines block.
    """
    if split not in {"train", "test"}:
        raise ValueError(f"D1 has only 'train' and 'test' splits, not {split!r}")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("SetFit/toxic_conversations", f"{split}.jsonl", repo_type="dataset")
    return read_toxic_conversations(path, sample=sample, seed=seed)


def split_d1(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_size: int | None = D1_TRAIN_SIZE,
    val_size: int = D1_VAL_SIZE,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make our D1 train and validation sets from the official train file.

    D1 has no validation split, so we hold one out of train. The validation set
    stays the same whatever `train_size` is, so models trained on different
    amounts of data are compared on the same messages. Texts that also appear in
    the official test file are dropped, so the final test stays unseen.
    """
    data = deduplicate(train)
    data = data[~data["text"].str.lower().isin(set(test["text"].str.lower()))]
    rest, val = train_test_split(data, test_size=val_size, stratify=data["label"], random_state=seed)
    if train_size is not None and train_size < len(rest):
        rest, _ = train_test_split(rest, train_size=train_size, stratify=rest["label"], random_state=seed)
    return rest.reset_index(drop=True), val.reset_index(drop=True)


def load_d1_train_val(
    train_size: int | None = D1_TRAIN_SIZE, val_size: int = D1_VAL_SIZE, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download D1 if needed and return the shared train and validation sets.

    Every toxicity model uses this function, so all of them see the same data.
    """
    return split_d1(load_toxic_conversations("train"), load_toxic_conversations("test"), train_size, val_size, seed)


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


def load_d3_splits(root: str | Path = "data/raw/gaslighting") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return D3's official train, validation and test sets, both classes in each.

    Train is deduplicated, and train texts that repeat in validation or test are
    dropped, so no message is scored on data it was trained on. Validation and
    test stay exactly as published.
    """
    root = Path(root)
    splits = [
        load_gaslighting([root / f"{prefix}_{split}.json" for prefix in ("gaslighting", "anti_gaslighting")])
        for split in ("train", "validation", "test")
    ]
    train, val, test = splits
    held_out = set(pd.concat([val["text"], test["text"]]).str.lower())
    train = deduplicate(train)
    train = train[~train["text"].str.lower().isin(held_out)].reset_index(drop=True)
    return train, val, test


def load_stress_test(path: str | Path, part: str = "dev") -> pd.DataFrame:
    """Load our own dating-style test set.

    `part="dev"` (the default) returns messages S001 to S120, whose model errors we
    have already seen. `part="final"` returns S121 onwards, which we score once, at
    the end. `part="all"` returns both, for labelling only. The default keeps every
    training script away from the final messages.
    """
    if part not in {"dev", "final", "all"}:
        raise ValueError(f"part must be 'dev', 'final' or 'all', not {part!r}")
    data = _read_csv(path)
    validate_required_columns(data, STRESS_COLUMNS)
    validate_binary_labels(data["label"])
    data["text"] = data["text"].map(clean_text)
    number = data["id"].str.extract(r"(\d+)", expand=False).astype(int)
    if part == "dev":
        data = data[number <= DEV_LAST_ID]
    elif part == "final":
        data = data[number > DEV_LAST_ID]
    return data.reset_index(drop=True)


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
