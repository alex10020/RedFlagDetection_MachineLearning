"""Two people label the dating messages independently; we measure how often they agree.

1. Make one blind sheet per labeller (no labels, no categories, own random order):
       python -m red_flag_detector.labelling sheets
2. Each labeller fills the `label` column of their own sheet (1 harmful, 0 not harmful)
   without talking to the other. Notes are optional.
3. Measure agreement and list the disagreements:
       python -m red_flag_detector.labelling kappa
4. The two labellers discuss each disagreement and write the outcome in the
   `agreed_label` column of data/labelling/disagreements.csv. Then:
       python -m red_flag_detector.labelling apply
   writes the final labels into data/dating_stress_test.csv with status `reviewed`.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .data import load_stress_test

STRESS = Path("data/dating_stress_test.csv")
SHEETS = Path("data/labelling")
SEEDS = {"A": 1, "B": 2}


def make_sheets(stress_path: Path = STRESS, out_dir: Path = SHEETS) -> list[Path]:
    """Write one blind sheet per labeller. Refuses to overwrite a sheet someone may have filled."""
    stress = load_stress_test(stress_path, part="all")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, seed in SEEDS.items():
        path = out_dir / f"labeller_{name}.csv"
        if path.exists():
            raise FileExistsError(f"{path} already exists. Delete it yourself if you really want a fresh sheet.")
        # Each labeller sees a different order, so neither copies the other's rhythm or fatigue.
        sheet = stress[["id", "text"]].sample(frac=1, random_state=seed).assign(label="", notes="")
        sheet.to_csv(path, index=False)
        paths.append(path)
    return paths


def read_sheet(path: Path, ids: pd.Series) -> pd.DataFrame:
    """Read a filled sheet and check that it labels every message with 0 or 1."""
    sheet = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
    missing = sorted(set(ids) - set(sheet["id"]))
    if missing:
        raise ValueError(f"{path} has no row for {len(missing)} messages, e.g. {missing[:3]}")
    sheet = sheet[sheet["id"].isin(set(ids))]
    bad = sheet[~sheet["label"].astype(str).str.strip().isin({"0", "1"})]
    if len(bad):
        raise ValueError(f"{path}: {len(bad)} rows have no 0/1 label, e.g. ids {bad['id'].head(3).tolist()}")
    return sheet.assign(label=sheet["label"].astype(str).str.strip().astype(int)).set_index("id")


def agreement(sheet_a: pd.DataFrame, sheet_b: pd.DataFrame, stress: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Return Cohen's kappa and agreement rates, plus one row per disagreement."""
    data = stress.set_index("id")[["text", "category", "label"]].rename(columns={"label": "draft_label"})
    data["label_a"] = sheet_a["label"]
    data["label_b"] = sheet_b["label"]
    data["notes_a"] = sheet_a.get("notes", "")
    data["notes_b"] = sheet_b.get("notes", "")
    same = data["label_a"] == data["label_b"]
    report = {
        "messages": len(data),
        "cohen_kappa": round(float(cohen_kappa_score(data["label_a"], data["label_b"])), 3),
        "raw_agreement": round(float(same.mean()), 3),
        "agreement_by_category": same.groupby(data["category"]).mean().round(2).to_dict(),
        # How often each person agrees with the AI draft; the draft is not ground truth.
        "labeller_a_vs_draft": round(float((data["label_a"] == data["draft_label"]).mean()), 3),
        "labeller_b_vs_draft": round(float((data["label_b"] == data["draft_label"]).mean()), 3),
    }
    disagreements = data[~same].reset_index().assign(agreed_label="")
    return report, disagreements[["id", "category", "text", "label_a", "label_b", "notes_a", "notes_b", "agreed_label"]]


def apply_labels(stress_path: Path, sheet_a: pd.DataFrame, sheet_b: pd.DataFrame, disagreements_path: Path) -> pd.DataFrame:
    """Write the final labels: the shared label where both agree, the agreed one where they did not."""
    raw = pd.read_csv(stress_path, dtype={"id": str})
    settled = pd.read_csv(disagreements_path, dtype={"id": str}, keep_default_na=False).set_index("id")["agreed_label"].astype(str).str.strip()
    final = {}
    for message_id in raw["id"]:
        a, b = sheet_a.at[message_id, "label"], sheet_b.at[message_id, "label"]
        if a == b:
            final[message_id] = a
        elif settled.get(message_id, "") in {"0", "1"}:
            final[message_id] = int(settled[message_id])
        else:
            raise ValueError(f"Message {message_id}: the labellers disagree and agreed_label is empty in {disagreements_path}")
    raw["label"] = raw["id"].map(final)
    raw["label_status"] = "reviewed"
    raw.to_csv(stress_path, index=False)
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["sheets", "kappa", "apply"])
    parser.add_argument("--stress", type=Path, default=STRESS)
    parser.add_argument("--dir", type=Path, default=SHEETS)
    args = parser.parse_args()
    if args.step == "sheets":
        for path in make_sheets(args.stress, args.dir):
            print(f"Wrote {path}")
        return
    stress = load_stress_test(args.stress, part="all")
    sheet_a = read_sheet(args.dir / "labeller_A.csv", stress["id"])
    sheet_b = read_sheet(args.dir / "labeller_B.csv", stress["id"])
    if args.step == "kappa":
        report, disagreements = agreement(sheet_a, sheet_b, stress)
        disagreements.to_csv(args.dir / "disagreements.csv", index=False)
        (args.dir / "agreement.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        print(f"{len(disagreements)} disagreements written to {args.dir / 'disagreements.csv'}")
    else:
        apply_labels(args.stress, sheet_a, sheet_b, args.dir / "disagreements.csv")
        print(f"Final labels written to {args.stress}")


if __name__ == "__main__":
    main()
