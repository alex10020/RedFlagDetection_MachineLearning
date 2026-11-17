import pandas as pd
import pytest

from red_flag_detector.data import (
    JIGSAW_TYPES,
    clean_text,
    deduplicate,
    load_gaslighting,
    load_jigsaw,
    load_jigsaw_test,
    load_stress_test,
    make_splits,
    read_toxic_conversations,
    load_d3_splits,
    split_d1,
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


def test_jigsaw_test_joins_labels_and_drops_unscored_rows(tmp_path):
    pd.DataFrame({"id": ["a", "b", "c"], "comment_text": ["fine", "you idiot", "unscored"]}).to_csv(tmp_path / "test.csv", index=False)
    zero = {name: 0 for name in JIGSAW_TYPES}
    labels = [{"id": "a", **zero}, {"id": "b", **zero, "insult": 1}, {"id": "c", **{name: -1 for name in JIGSAW_TYPES}}]
    pd.DataFrame(labels).to_csv(tmp_path / "test_labels.csv", index=False)
    data = load_jigsaw_test(tmp_path / "test.csv", tmp_path / "test_labels.csv")
    assert data["text"].tolist() == ["fine", "you idiot"]
    assert data["label"].tolist() == [0, 1]


def test_stress_test_keeps_final_messages_out_of_dev(tmp_path):
    rows = [{"id": f"S{i:03d}", "text": "hi", "label": 0, "category": "c", "topic": "harmless", "label_status": "x"} for i in (1, 120, 121, 200)]
    path = tmp_path / "stress.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    assert load_stress_test(path)["id"].tolist() == ["S001", "S120"]
    assert load_stress_test(path, part="final")["id"].tolist() == ["S121", "S200"]
    assert len(load_stress_test(path, part="all")) == 4


def test_missing_file_points_to_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        load_jigsaw(tmp_path / "nope.csv")


def test_toxic_conversations_reads_jsonl_and_samples_by_class(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [{"text": f"  msg\n{i} ", "label": int(i % 5 == 0), "label_text": "x"} for i in range(100)]
    pd.DataFrame(rows).to_json(path, orient="records", lines=True)
    full = read_toxic_conversations(path)
    assert list(full.columns) == ["text", "label"]
    assert full["text"][0] == "msg 0"
    sample = read_toxic_conversations(path, sample=50)
    assert len(sample) == 50
    assert sample["label"].mean() == pytest.approx(0.2)


def test_toxic_conversations_missing_file_points_to_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        read_toxic_conversations(tmp_path / "nope.jsonl")


def test_deduplicate_ignores_case_and_drops_conflicts():
    data = pd.DataFrame({"text": ["Hi", "hi", "both", "both", "solo"], "label": [0, 0, 0, 1, 1]})
    assert deduplicate(data)["text"].tolist() == ["Hi", "solo"]


def test_splits_keep_class_share_and_do_not_overlap():
    train, val, test = make_splits(_labelled(), val_size=0.2, test_size=0.2)
    assert (len(train), len(val), len(test)) == (60, 20, 20)
    for part in (train, val, test):
        assert part["label"].mean() == pytest.approx(0.2)
    assert validate_no_text_overlap(train, val, test)


def test_split_d1_drops_test_texts_and_keeps_validation_fixed():
    train = pd.DataFrame(
        {"text": [f"msg {i}" for i in range(1000)] + ["MSG 0", "in test"], "label": [int(i % 5 == 0) for i in range(1000)] + [1, 0]}
    )
    test = pd.DataFrame({"text": ["In Test"], "label": [0]})
    small, val_small = split_d1(train, test, train_size=200, val_size=100)
    full, val_full = split_d1(train, test, train_size=None, val_size=100)
    assert len(small) == 200 and len(val_small) == 100
    assert val_small["text"].tolist() == val_full["text"].tolist()
    assert len(full) == 900  # 1002 rows, minus the "MSG 0" duplicate, the test text and 100 validation rows
    for part in (small, full, val_full):
        assert "in test" not in part["text"].str.lower().tolist()
        assert part["label"].mean() == pytest.approx(0.2, abs=0.01)
    assert validate_no_text_overlap(full, val_full)


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


def test_d3_splits_drop_train_copies_of_held_out_texts(tmp_path):
    texts = {
        "train": [("you imagined it", "gaslighting"), ("You did well", "safe"), ("fresh start", "safe")],
        "validation": [("you did well", "safe")],
        "test": [("never said that", "gaslighting")],
    }
    for split, rows in texts.items():
        for prefix, category in (("gaslighting", "gaslighting"), ("anti_gaslighting", "safe")):
            chosen = [_d3_row(i, "assistant", text, cat) for i, (text, cat) in enumerate(rows) if cat == category]
            pd.DataFrame(chosen, columns=list(_d3_row(0, "", "", "").keys())).to_json(
                tmp_path / f"{prefix}_{split}.json", orient="records"
            )
    train, val, test = load_d3_splits(tmp_path)
    assert sorted(train["text"]) == ["fresh start", "you imagined it"]
    assert val["text"].tolist() == ["you did well"]
    assert test["label"].tolist() == [1]


def test_gaslighting_rejects_unknown_category(tmp_path):
    path = tmp_path / "d3.json"
    pd.DataFrame([_d3_row(1, "assistant", "hi", "other")]).to_json(path, orient="records")
    with pytest.raises(ValueError, match="other"):
        load_gaslighting([path])
