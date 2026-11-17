import numpy as np
import pandas as pd

from red_flag_detector.errors import draft_cause, pick_errors


def test_pick_errors_keeps_most_confident_mistakes_on_each_side():
    data = pd.DataFrame({"text": list("abcdef"), "label": [0, 0, 0, 1, 1, 1]})
    scores = np.array([0.9, 0.7, 0.1, 0.2, 0.05, 0.8])
    errors = pick_errors(data, scores, threshold=0.5, per_side=1)
    assert errors["text"].tolist() == ["a", "e"]
    assert errors["error"].tolist() == ["false positive", "false negative"]


def test_draft_cause_uses_dating_category_first():
    assert draft_cause("i will find you", 1, "toxic", "threat") == "threat without insult words"
    assert draft_cause("loved your photos", 0, "gaslighting", "friendly_flirt") == "harmless chat in a style the training data lacks"


def test_draft_cause_rules_for_validation_messages():
    assert draft_cause("Muslims voted today", 0, "toxic") == "identity word"
    assert draft_cause("you are a waste of space", 1, "toxic") == "toxic without rude words"
    assert draft_cause("Overpaid, childish, Morons", 0, "toxic") == "rude or insulting words, labelled not toxic"
    assert draft_cause("How did that feel?", 0, "gaslighting") == "question in a safe chat"
    assert draft_cause("My week was long.", 1, "gaslighting") == "turn reads like the user"
