import pandas as pd
import pytest

from red_flag_detector.validation import (
    validate_binary_labels,
    validate_no_text_overlap,
    validate_required_columns,
)


def test_required_columns_are_present():
    df = pd.DataFrame({"text": ["hi"], "label": [0]})
    assert validate_required_columns(df, ["text", "label"]) is True


def test_missing_column_raises():
    with pytest.raises(ValueError, match="label"):
        validate_required_columns(pd.DataFrame({"text": ["hi"]}), ["text", "label"])


def test_labels_must_be_binary():
    with pytest.raises(ValueError):
        validate_binary_labels(pd.Series([0, 1, 2]))


def test_overlap_between_splits_raises():
    a = pd.DataFrame({"text": ["same", "x"]})
    b = pd.DataFrame({"text": ["same", "y"]})
    with pytest.raises(ValueError, match="leakage"):
        validate_no_text_overlap(a, b)
