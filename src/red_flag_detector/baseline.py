"""Model 1: TF-IDF word counts and logistic regression.

Run with:
    python -m red_flag_detector.baseline --dataset d1                 # shared 100,000-message D1 sample
    python -m red_flag_detector.baseline --dataset d1 --d1-train-size all
    python -m red_flag_detector.baseline --dataset d2 --data data/raw/jigsaw_train.csv
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import D1_TRAIN_SIZE, D1_VAL_SIZE, deduplicate, load_d1_train_val, load_jigsaw, load_stress_test, make_splits
from .evaluate import best_f1_threshold, binary_metrics, bootstrap_interval, category_report
from .validation import validate_no_text_overlap

C_GRID = (0.1, 1.0, 10.0)
SEED = 42


def build_baseline(c: float = 1.0) -> Pipeline:
    """Build the pipeline. Fitting it on train data keeps validation words out of the vocabulary."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, max_features=50_000, sublinear_tf=True),
            ),
            ("logreg", LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=SEED)),
        ]
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_model(train: pd.DataFrame, val: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Fit one model per C and keep the one with the best validation F1 at its best threshold."""
    trials = []
    best = None
    for c in C_GRID:
        started = time.perf_counter()
        model = build_baseline(c).fit(train["text"], train["label"])
        seconds = time.perf_counter() - started
        scores = model.predict_proba(val["text"])[:, 1]
        threshold = best_f1_threshold(val["label"], scores)
        metrics = binary_metrics(val["label"], scores, threshold)
        trials.append({"C": c, "threshold": threshold, "train_seconds": round(seconds, 1), **metrics})
        if best is None or metrics["f1"] > best["metrics"]["f1"]:
            best = {"model": model, "c": c, "threshold": threshold, "metrics": metrics, "train_seconds": round(seconds, 1)}
    return best, trials


def stress_summary(stress: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict:
    """Score the dating test as a whole and per topic against the harmless messages."""
    flags = (scores >= threshold).astype(int)
    harmless = (stress["label"] == 0).to_numpy()
    groups = {
        "all": np.ones(len(stress), dtype=bool),
        "toxic_vs_harmless": ((stress["topic"] == "toxic") | harmless).to_numpy(),
        "gaslighting_vs_harmless": ((stress["topic"] == "gaslighting") | harmless).to_numpy(),
    }
    summary = {}
    for name, rows in groups.items():
        summary[name] = binary_metrics(stress["label"][rows], scores[rows], threshold)
        summary[name]["messages"] = int(rows.sum())
        summary[name]["f1_95ci"] = bootstrap_interval(stress["label"][rows], flags[rows])
    summary["false_positive_rate"] = float(flags[harmless].mean())
    return summary


def save_stress_outputs(stress: pd.DataFrame, scores: np.ndarray, threshold: float, by_category: Path, errors: Path) -> None:
    """Write the flag rate per category and every message the model got wrong."""
    flags = (scores >= threshold).astype(int)
    category_report(stress, scores, threshold).to_csv(by_category, index=False)
    stress.assign(score=np.round(scores, 3), flagged=flags).query("flagged != label").drop(
        columns="label_status"
    ).sort_values("category").to_csv(errors, index=False)


def _split_info(parts: dict[str, pd.DataFrame]) -> dict:
    return {name: {"rows": len(part), "toxic_share": round(float(part["label"].mean()), 4)} for name, part in parts.items()}


def run_d2(data_path: Path, stress_path: Path, results_dir: Path, model_path: Path) -> dict:
    """Train on our 64/16/20 split of Jigsaw train.csv. This was the first stand-in baseline."""
    data = load_jigsaw(data_path)
    rows_loaded = len(data)
    data = deduplicate(data)[["text", "label"]]
    train, val, test = make_splits(data, val_size=0.16, test_size=0.20, seed=SEED)
    validate_no_text_overlap(train, val, test)

    best, trials = select_model(train, val)
    model, threshold = best["model"], best["threshold"]

    test_scores = model.predict_proba(test["text"])[:, 1]
    stress = load_stress_test(stress_path)
    stress_scores = model.predict_proba(stress["text"])[:, 1]

    results_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "trained_on": "D2 Jigsaw train.csv"}, model_path, compress=3)
    save_stress_outputs(
        stress, stress_scores, threshold,
        results_dir / "dating_stress_test_by_category.csv", results_dir / "dating_stress_test_errors.csv",
    )

    report = {
        "dataset": "D2 Jigsaw train.csv, split internally",
        "dataset_sha256": _sha256(data_path),
        "rows_loaded": rows_loaded,
        "rows_after_dedup": len(data),
        "splits": _split_info({"train": train, "val": val, "test": test}),
        "c_trials": trials,
        "chosen_c": best["c"],
        "threshold": threshold,
        "test": binary_metrics(test["label"], test_scores, threshold),
        "test_at_threshold_0.5": binary_metrics(test["label"], test_scores, 0.5),
        "majority_class_test_accuracy": float(1 - test["label"].mean()),
        "dating_stress_test": stress_summary(stress, stress_scores, threshold),
    }
    (results_dir / "baseline_metrics.json").write_text(json.dumps(report, indent=2))
    return report


def run_d1(train_size: int | None, stress_path: Path, results_dir: Path, model_path: Path) -> dict:
    """Train on D1 train, tune on our held-out D1 validation set, score the dating dev set.

    The official D1 test file is not scored here: we run it once, at the end, for all models.
    """
    train, val = load_d1_train_val(train_size=train_size, val_size=D1_VAL_SIZE, seed=SEED)
    validate_no_text_overlap(train, val)

    best, trials = select_model(train, val)
    model, threshold = best["model"], best["threshold"]

    stress = load_stress_test(stress_path)
    stress_scores = model.predict_proba(stress["text"])[:, 1]

    results_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "trained_on": f"D1 train, {len(train)} messages"}, model_path, compress=3)
    save_stress_outputs(stress, stress_scores, threshold, results_dir / "dating_by_category.csv", results_dir / "dating_errors.csv")

    report = {
        "dataset": "D1 SetFit/toxic_conversations train.jsonl, deduplicated, official test texts removed",
        "splits": _split_info({"train": train, "val": val}),
        "c_trials": trials,
        "chosen_c": best["c"],
        "threshold": threshold,
        "train_seconds": best["train_seconds"],
        "model_mb": round(model_path.stat().st_size / 1e6, 2),
        "validation": best["metrics"],
        "validation_at_threshold_0.5": binary_metrics(val["label"], model.predict_proba(val["text"])[:, 1], 0.5),
        "dating_dev_set": stress_summary(stress, stress_scores, threshold),
    }
    (results_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["d1", "d2"], default="d1")
    parser.add_argument("--data", type=Path, default=Path("data/raw/jigsaw_train.csv"), help="D2 only")
    parser.add_argument("--d1-train-size", default=str(D1_TRAIN_SIZE), help="number of D1 training messages, or 'all'")
    parser.add_argument("--stress", type=Path, default=Path("data/dating_stress_test.csv"))
    parser.add_argument("--results", type=Path, help="defaults to results/ (D2) or results/d1_baseline_<size>/ (D1)")
    parser.add_argument("--model", type=Path, help="defaults to models/baseline_v0.joblib (D2) or models/baseline_d1_<size>.joblib")
    args = parser.parse_args()

    if args.dataset == "d2":
        report = run_d2(args.data, args.stress, args.results or Path("results"), args.model or Path("models/baseline_v0.joblib"))
    else:
        train_size = None if args.d1_train_size == "all" else int(args.d1_train_size)
        tag = "all" if train_size is None else f"{train_size // 1000}k"
        report = run_d1(
            train_size,
            args.stress,
            args.results or Path(f"results/d1_baseline_{tag}"),
            args.model or Path(f"models/baseline_d1_{tag}.joblib"),
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
