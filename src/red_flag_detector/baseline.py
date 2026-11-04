"""Model 1: TF-IDF word counts and logistic regression.

Run with:
    python -m red_flag_detector.baseline --data data/raw/jigsaw_train.csv
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

from .data import deduplicate, load_jigsaw, load_stress_test, make_splits
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


def run(data_path: Path, stress_path: Path, results_dir: Path, model_path: Path) -> dict:
    data = load_jigsaw(data_path)
    rows_loaded = len(data)
    data = deduplicate(data)[["text", "label"]]
    train, val, test = make_splits(data, val_size=0.16, test_size=0.20, seed=SEED)
    validate_no_text_overlap(train, val, test)

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
            best = {"model": model, "c": c, "threshold": threshold, "metrics": metrics}
    model, threshold = best["model"], best["threshold"]

    test_scores = model.predict_proba(test["text"])[:, 1]
    test_metrics = binary_metrics(test["label"], test_scores, threshold)
    test_at_half = binary_metrics(test["label"], test_scores, 0.5)
    test_accuracy_majority = float(1 - test["label"].mean())

    stress = load_stress_test(stress_path)
    stress_scores = model.predict_proba(stress["text"])[:, 1]
    stress_flags = (stress_scores >= threshold).astype(int)
    core = (stress["scope"] == "core").to_numpy()
    harmless = (stress["label"] == 0).to_numpy()
    summary = {
        "all": binary_metrics(stress["label"], stress_scores, threshold),
        "core": binary_metrics(stress["label"][core], stress_scores[core], threshold),
    }
    summary["all"]["f1_95ci"] = bootstrap_interval(stress["label"], stress_flags)
    summary["core"]["f1_95ci"] = bootstrap_interval(stress["label"][core], stress_flags[core])
    summary["false_positive_rate"] = float(stress_flags[harmless].mean())

    results_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "trained_on": "D2 Jigsaw train.csv"}, model_path, compress=3)
    category_report(stress, stress_scores, threshold).to_csv(results_dir / "dating_stress_test_by_category.csv", index=False)
    stress.assign(score=np.round(stress_scores, 3), flagged=stress_flags).query("flagged != label").drop(
        columns="label_status"
    ).sort_values("category").to_csv(results_dir / "dating_stress_test_errors.csv", index=False)

    report = {
        "dataset": "D2 Jigsaw train.csv, split internally",
        "dataset_sha256": _sha256(data_path),
        "rows_loaded": rows_loaded,
        "rows_after_dedup": len(data),
        "splits": {name: {"rows": len(part), "toxic_share": round(float(part["label"].mean()), 4)} for name, part in zip(("train", "val", "test"), (train, val, test))},
        "c_trials": trials,
        "chosen_c": best["c"],
        "threshold": threshold,
        "test": test_metrics,
        "test_at_threshold_0.5": test_at_half,
        "majority_class_test_accuracy": test_accuracy_majority,
        "dating_stress_test": summary,
    }
    (results_dir / "baseline_metrics.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/jigsaw_train.csv"))
    parser.add_argument("--stress", type=Path, default=Path("data/dating_stress_test.csv"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--model", type=Path, default=Path("models/baseline_v0.joblib"))
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.stress, args.results, args.model), indent=2))


if __name__ == "__main__":
    main()
