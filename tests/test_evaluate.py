import numpy as np
import pandas as pd
import pytest

from red_flag_detector.baseline import build_baseline
from red_flag_detector.evaluate import best_f1_threshold, binary_metrics, bootstrap_interval, category_report, paired_bootstrap_difference


def test_always_not_toxic_scores_zero_f1():
    assert binary_metrics([0, 0, 0, 1], [0.1, 0.1, 0.1, 0.1], threshold=0.5)["f1"] == 0


def test_best_threshold_separates_clean_scores():
    threshold = best_f1_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold)["f1"] == 1


def test_paired_bootstrap_finds_a_clear_winner_and_a_tie():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 200)
    good = np.where(y == 1, 0.9, 0.1) + rng.normal(0, 0.05, len(y))
    bad = rng.random(len(y))
    win = paired_bootstrap_difference(y, good, 0.5, bad, 0.5, metric="roc_auc", n_resamples=200)
    assert win["ci_low"] > 0.3
    tie = paired_bootstrap_difference(y, good, 0.5, good, 0.5, metric="f1", n_resamples=200)
    assert tie["difference"] == 0 and tie["ci_low"] == 0 == tie["ci_high"]


def test_bootstrap_interval_contains_the_metric():
    low, high = bootstrap_interval([0, 1] * 20, [0, 1] * 15 + [1, 0] * 5, n_resamples=200)
    assert low <= 0.75 <= high


def test_category_report_counts_flags():
    stress = pd.DataFrame({"label": [1, 1, 0], "topic": ["toxic", "toxic", "harmless"], "category": ["threat", "threat", "flirt"]})
    table = category_report(stress, [0.9, 0.1, 0.9], threshold=0.5).set_index("category")
    assert table.loc["threat", "flag_rate"] == pytest.approx(0.5)
    assert table.loc["flirt", "flagged"] == 1


def test_baseline_learns_a_simple_pattern():
    texts = ["you are horrible"] * 10 + ["see you at noon"] * 10
    labels = [1] * 10 + [0] * 10
    model = build_baseline()
    model.named_steps["tfidf"].min_df = 1
    model.fit(texts, labels)
    assert model.predict_proba(["you are horrible"])[0, 1] > model.predict_proba(["see you at noon"])[0, 1]
