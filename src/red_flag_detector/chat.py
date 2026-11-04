"""Turn a pasted chat into a scored report. Works with any model that returns scores."""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

Scorer = Callable[[Sequence[str]], np.ndarray]

# Optional WhatsApp-style prefixes: "[12/03/24, 21:14] " or "12/03/24, 21:14 - "
_TIMESTAMP = r"(?:\[[^\]]*\d[^\]]*\]\s*|\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4},?\s+\d{1,2}[:.]\d{2}(?:\s?[APap][Mm])?\s*-\s*)?"
_LINE = re.compile(rf"^{_TIMESTAMP}(?P<sender>[^\d\s:][^:]{{0,29}}):\s+(?P<text>.+)$")

# Default cut-off of the class-weighted model. On D2 test comments it trades precision for recall.
WATCH_THRESHOLD = 0.5
MIN_WORD_EFFECT = 0.05


@dataclass
class ScoredMessage:
    sender: str
    text: str
    score: float
    level: str
    word_effects: list[tuple[str, float]]


def parse_chat(chat: str) -> list[tuple[str, str]]:
    """Split a chat into (sender, message). A line with no sender continues the last message."""
    messages: list[tuple[str, str]] = []
    for line in chat.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _LINE.match(line)
        if match:
            messages.append((match["sender"].strip(), match["text"].strip()))
        elif messages:
            sender, text = messages[-1]
            messages[-1] = (sender, f"{text} {line}")
        else:
            messages.append(("Unknown", line))
    return messages


def level_for(score: float, threshold: float) -> str:
    if score >= threshold:
        return "red flag"
    if score >= WATCH_THRESHOLD:
        return "worth a look"
    return "no flag"


def word_effects(text: str, score: float, scorer: Scorer) -> list[tuple[str, float]]:
    """Rescore the message without each word. The score drop is that word's effect."""
    words = text.split()
    if len(words) < 2:
        return [(text, 0.0)]
    without_each = [" ".join(words[:i] + words[i + 1 :]) for i in range(len(words))]
    scores = scorer(without_each)
    return [(word, float(score - other)) for word, other in zip(words, scores)]


def score_chat(chat: str, scorer: Scorer, threshold: float) -> list[ScoredMessage]:
    """Score each message. Explain only the flagged ones, which keeps long chats fast."""
    parsed = parse_chat(chat)
    if not parsed:
        return []
    scores = scorer([text for _, text in parsed])
    report = []
    for (sender, text), raw in zip(parsed, scores):
        score = float(raw)
        level = level_for(score, threshold)
        effects = word_effects(text, score, scorer) if level != "no flag" else [(word, 0.0) for word in text.split()]
        report.append(ScoredMessage(sender, text, score, level, effects))
    return report


def summarise_senders(report: Sequence[ScoredMessage]) -> list[dict]:
    """Count messages and flags per sender, most flags first."""
    senders: dict[str, dict] = {}
    for message in report:
        row = senders.setdefault(message.sender, {"sender": message.sender, "messages": 0, "red_flags": 0, "worth_a_look": 0})
        row["messages"] += 1
        row["red_flags"] += message.level == "red flag"
        row["worth_a_look"] += message.level == "worth a look"
    return sorted(senders.values(), key=lambda row: (-row["red_flags"], -row["worth_a_look"], row["sender"]))
