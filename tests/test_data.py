import pandas as pd
import pytest

from red_flag_detector.data import (
    JIGSAW_TYPES,
    clean_text,
    deduplicate,
    load_gaslighting,
    load_jigsaw,
    make_splits,
)
from red_flag_detector.validation import validate_no_text_overlap


def _labelled(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame({"text": [f"message {i}" for i in range(n)], "label": [i % 5 == 0 for i in range(n)]}).astype(
        {"label": int}
    )


def test_clean_text_collapses_whitespace_and_keeps_case():
    assert clean_text("  HEY\n\nyou \t there ") == "HEY you there"


def test_jigsaw_counts_any_type_as_toxic(tmp_path):
    row = {name: 0 for name in JIGSAW_TYPES}
    rows = [
        {"comment_text": "fine", **row},
        {"comment_text": "threat only", **row, "threat": 1},
    ]
    path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    assert load_jigsaw(path)["label"].tolist() == [0, 1]


def test_missing_file_points_to_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        load_jigsaw(tmp_path / "nope.csv")


def test_deduplicate_ignores_case_and_drops_conflicts():
    data = pd.DataFrame({"text": ["Hi", "hi", "both", "both", "solo"], "label": [0, 0, 0, 1, 1]})
    assert deduplicate(data)["text"].tolist() == ["Hi", "solo"]


def test_splits_keep_class_share_and_do_not_overlap():
    train, val, test = make_splits(_labelled(), val_size=0.2, test_size=0.2)
    assert (len(train), len(val), len(test)) == (60, 20, 20)
    for part in (train, val, test):
        assert part["label"].mean() == pytest.approx(0.2)
    assert validate_no_text_overlap(train, val, test)


def _d3_row(chat, role, text, category):
    return {"id": chat, "turn": 0, "speaker": "x", "speaker role": role, "internal thought": "secret", "utterance": text, "category": category}


def test_gaslighting_keeps_assistant_turns_and_drops_thoughts(tmp_path):
    path = tmp_path / "d3.json"
    rows = [
        _d3_row(1, "user", "i feel low", "gaslighting"),
        _d3_row(1, "assistant", "you imagined it", "gaslighting"),
        _d3_row(1, "assistant", "you did well", "safe"),
    ]
    pd.DataFrame(rows).to_json(path, orient="records")
    data = load_gaslighting([path])
    assert data["text"].tolist() == ["you imagined it", "you did well"]
    assert data["label"].tolist() == [1, 0]
    assert "internal thought" not in data.columns


def test_gaslighting_rejects_unknown_category(tmp_path):
    path = tmp_path / "d3.json"
    pd.DataFrame([_d3_row(1, "assistant", "hi", "other")]).to_json(path, orient="records")
    with pytest.raises(ValueError, match="other"):
        load_gaslighting([path])
