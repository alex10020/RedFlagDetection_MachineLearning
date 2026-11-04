import numpy as np

from red_flag_detector.app import render_report
from red_flag_detector.chat import level_for, parse_chat, score_chat, summarise_senders


def fake_scorer(texts):
    return np.array([0.95 if "awful" in text else 0.1 for text in texts])


def test_parse_plain_and_whatsapp_lines():
    chat = "Sam: hi there\n[12/03/24, 21:14] Maya: see https://x.org: ok\n12/03/24, 21:15 - Sam: bye"
    assert parse_chat(chat) == [("Sam", "hi there"), ("Maya", "see https://x.org: ok"), ("Sam", "bye")]


def test_line_without_sender_continues_last_message():
    assert parse_chat("Sam: hi\nhow are you") == [("Sam", "hi how are you")]


def test_first_line_without_sender_is_unknown():
    assert parse_chat("hello") == [("Unknown", "hello")]


def test_time_is_not_a_sender():
    assert parse_chat("Sam: meet at\n10:30 is good") == [("Sam", "meet at 10:30 is good")]


def test_levels():
    assert [level_for(s, 0.7) for s in (0.8, 0.6, 0.2)] == ["red flag", "worth a look", "no flag"]


def test_word_effect_finds_the_word_behind_the_flag():
    report = score_chat("Sam: you are awful", fake_scorer, threshold=0.7)
    effects = dict(report[0].word_effects)
    assert report[0].level == "red flag"
    assert effects["awful"] > 0.5 and effects["you"] == 0


def test_summary_puts_most_flags_first():
    report = score_chat("A: fine\nB: awful\nB: fine", fake_scorer, threshold=0.7)
    assert summarise_senders(report)[0] == {"sender": "B", "messages": 2, "red_flags": 1, "worth_a_look": 0}


def test_report_escapes_chat_text():
    report = score_chat("<b>Sam</b>: <script>alert(1)</script> awful", fake_scorer, threshold=0.7)
    page = render_report(report)
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert "<b>Sam</b>" not in page.replace("<b>&lt;", "")


def test_empty_chat_shows_prompt():
    assert "Paste a chat" in render_report(score_chat("  \n ", fake_scorer, threshold=0.7))
