"""Metrics for the toxic class. Accuracy stays out: always saying "not toxic" scores high."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score


def binary_metrics(y_true, scores, threshold: float) -> dict[str, float]:
    """Return toxic-class precision, recall, F1 and PR-AUC at a threshold."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)),
    }


def best_f1_threshold(y_true, scores) -> float:
    """Return the threshold with the highest F1. Call this on validation data only."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator, out=np.zeros_like(denominator), where=denominator > 0)
    return float(thresholds[int(np.argmax(f1))])


def bootstrap_interval(y_true, y_pred, metric=f1_score, n_resamples: int = 2000, seed: int = 42) -> tuple[float, float]:
    """Return a 95% percentile interval. Small test sets need one."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    values = [
        metric(y_true[index], y_pred[index], zero_division=0)
        for index in rng.integers(0, len(y_true), size=(n_resamples, len(y_true)))
    ]
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def paired_bootstrap_difference(
    y_true, scores_a, threshold_a: float, scores_b, threshold_b: float, metric: str = "f1",
    n_resamples: int = 1000, seed: int = 42,
) -> dict[str, float]:
    """Difference A minus B on the same messages, with a 95% percentile interval.

    Both models are scored on the same resampled messages each time, so the
    interval reflects only how much the gap between them depends on which
    messages happen to be in the test set. `metric` is "f1" (at each model's own
    threshold) or "roc_auc" (no threshold).
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true)
    a, b = np.asarray(scores_a), np.asarray(scores_b)

    def value(index, scores, threshold):
        if metric == "roc_auc":
            return roc_auc_score(y[index], scores[index])
        return f1_score(y[index], scores[index] >= threshold, zero_division=0)

    full = np.arange(len(y))
    observed = value(full, a, threshold_a) - value(full, b, threshold_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_resamples):
        index = rng.integers(0, len(y), size=len(y))
        if metric == "roc_auc" and y[index].min() == y[index].max():
            continue  # ROC-AUC needs both classes in the resample
        diffs.append(value(index, a, threshold_a) - value(index, b, threshold_b))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"difference": float(observed), "ci_low": float(low), "ci_high": float(high)}


def category_report(stress: pd.DataFrame, scores, threshold: float) -> pd.DataFrame:
    """Return the share of messages flagged in each category of the dating test."""
    flagged = np.asarray(scores) >= threshold
    table = (
        stress.assign(flagged=flagged)
        .groupby(["label", "topic", "category"], as_index=False)
        .agg(messages=("flagged", "size"), flagged=("flagged", "sum"))
    )
    table["flag_rate"] = table["flagged"] / table["messages"]
    return table
