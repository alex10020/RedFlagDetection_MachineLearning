import pandas as pd
import pytest

from red_flag_detector.baseline import build_baseline
from red_flag_detector.evaluate import best_f1_threshold, binary_metrics, bootstrap_interval, category_report


def test_always_not_toxic_scores_zero_f1():
    assert binary_metrics([0, 0, 0, 1], [0.1, 0.1, 0.1, 0.1], threshold=0.5)["f1"] == 0


def test_best_threshold_separates_clean_scores():
    threshold = best_f1_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold)["f1"] == 1


def test_bootstrap_interval_contains_the_metric():
    low, high = bootstrap_interval([0, 1] * 20, [0, 1] * 15 + [1, 0] * 5, n_resamples=200)
    assert low <= 0.75 <= high


def test_category_report_counts_flags():
    stress = pd.DataFrame({"label": [1, 1, 0], "scope": ["core"] * 3, "category": ["threat", "threat", "flirt"]})
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
