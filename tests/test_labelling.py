import pandas as pd
import pytest

from red_flag_detector.labelling import agreement, apply_labels, make_sheets, read_sheet


def _stress(tmp_path):
    rows = [
        {"id": f"S{i:03d}", "text": f"msg {i}", "label": i % 2, "category": "threat" if i % 2 else "friendly_flirt",
         "topic": "toxic" if i % 2 else "harmless", "label_status": "draft_ai"}
        for i in range(1, 7)
    ]
    path = tmp_path / "stress.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _fill(path, labels):
    sheet = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
    sheet["label"] = sheet["id"].map(labels)
    sheet.to_csv(path, index=False)


def test_sheets_hide_labels_shuffle_and_refuse_to_overwrite(tmp_path):
    stress = _stress(tmp_path)
    a, b = make_sheets(stress, tmp_path / "sheets")
    sheet_a, sheet_b = pd.read_csv(a), pd.read_csv(b)
    assert list(sheet_a.columns) == ["id", "text", "label", "notes"]
    assert sheet_a["label"].isna().all()
    assert sheet_a["id"].tolist() != sheet_b["id"].tolist()
    with pytest.raises(FileExistsError):
        make_sheets(stress, tmp_path / "sheets")


def test_read_sheet_rejects_missing_labels(tmp_path):
    stress = _stress(tmp_path)
    a, _ = make_sheets(stress, tmp_path / "sheets")
    with pytest.raises(ValueError, match="no 0/1 label"):
        read_sheet(a, pd.read_csv(stress)["id"])


def test_kappa_disagreements_and_apply(tmp_path):
    stress_path = _stress(tmp_path)
    a, b = make_sheets(stress_path, tmp_path / "sheets")
    labels_a = {"S001": 1, "S002": 0, "S003": 1, "S004": 0, "S005": 1, "S006": 0}
    labels_b = {**labels_a, "S006": 1}
    _fill(a, labels_a)
    _fill(b, labels_b)
    stress = pd.read_csv(stress_path, dtype={"id": str})
    sheet_a, sheet_b = read_sheet(a, stress["id"]), read_sheet(b, stress["id"])
    report, disagreements = agreement(sheet_a, sheet_b, stress)
    assert report["raw_agreement"] == pytest.approx(5 / 6, abs=1e-3)
    assert 0 < report["cohen_kappa"] < 1
    assert disagreements["id"].tolist() == ["S006"]

    path = tmp_path / "disagreements.csv"
    disagreements.to_csv(path, index=False)
    with pytest.raises(ValueError, match="agreed_label is empty"):
        apply_labels(stress_path, sheet_a, sheet_b, path)
    disagreements.assign(agreed_label=0).to_csv(path, index=False)
    final = apply_labels(stress_path, sheet_a, sheet_b, path)
    assert final.set_index("id")["label"].to_dict() == labels_a
    assert (final["label_status"] == "reviewed").all()
