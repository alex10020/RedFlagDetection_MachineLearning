import random
from pathlib import Path

import numpy as np
import pandas as pd

from red_flag_detector.app import render_report
from red_flag_detector.chat import parse_chat, score_chat
from red_flag_detector.generator import HARMFUL_LABELS, HARMLESS, SCENARIOS, generate_chat

DATING_CSV = Path(__file__).resolve().parents[1] / "data" / "dating_stress_test.csv"


def test_generated_chat_has_one_label_per_message():
    for seed in range(50):
        chat = generate_chat(random.Random(seed))
        parsed = parse_chat(chat.text)
        assert len(parsed) == len(chat.truths)
        assert len({sender for sender, _ in parsed}) == 2
        assert all(label == HARMLESS or label in HARMFUL_LABELS for label in chat.truths)


def test_generator_gives_different_chats():
    texts = {generate_chat(random.Random(seed)).text for seed in range(30)}
    assert len(texts) > 20


def test_no_demo_message_comes_from_the_dating_test_set():
    """The demo must not show or score any of our 200 dating messages, above all the final 80."""
    dating = set(pd.read_csv(DATING_CSV)["text"].str.lower().str.strip())
    demo = {wording.lower().strip() for scenario in SCENARIOS for _, wordings, _ in scenario.turns for wording in wordings}
    assert not demo & dating


def test_reveal_shows_caught_missed_and_false_alarms():
    chat = "Ana: you are an idiot\nBen: lovely weather\nAna: i know where you live\nBen: shut up you nerd lol"
    scores = {"you are an idiot": 0.9, "lovely weather": 0.1, "i know where you live": 0.2, "shut up you nerd lol": 0.8}
    scorer = lambda texts: np.array([scores.get(t, 0.1) for t in texts])
    report = score_chat(chat, scorer, threshold=0.7)
    page = render_report({"toxic": report}, ["insult", HARMLESS, "threat", HARMLESS], "Test")
    assert "Caught" in page and "Missed" in page and "False alarm" in page and "Correctly left alone" in page
    assert "1 / 2" in page and "Scenario: <b>Test</b>" in page


def test_report_without_matching_truths_has_no_scoreboard():
    report = score_chat("Ana: hello", lambda texts: np.array([0.1] * len(texts)), threshold=0.7)
    assert "The truth" not in render_report({"toxic": report}, [HARMLESS, HARMLESS])
