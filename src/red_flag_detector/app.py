"""Prototype v1: a toxic-language score and a gaslighting score for each message.

Run with: python -m red_flag_detector.app

Both scores come from our CNN models (models/cnn_d1.pt and models/cnn_d3plus.pt).
Without PyTorch installed, the app falls back to the TF-IDF baselines.
"""

import argparse
import html
from pathlib import Path

import gradio as gr
import joblib
import numpy as np

from .chat import MIN_WORD_EFFECT, ScoredMessage, Scorer, score_chat

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
TOPICS = {"toxic": "Toxic", "gaslighting": "Gaslighting"}
# The gaslighting CNN was trained on D3 plus everyday D1 messages, which cuts its false alarms on dating chat.
CNN_MODELS = {"toxic": MODELS_DIR / "cnn_d1.pt", "gaslighting": MODELS_DIR / "cnn_d3plus.pt"}
BASELINE_MODELS = {"toxic": MODELS_DIR / "baseline_d1_100k.joblib", "gaslighting": MODELS_DIR / "baseline_d3plus.joblib"}
LEVEL_ORDER = {"no flag": 0, "worth a look": 1, "red flag": 2}
# Shown as a grey number only: it never raises a flag, colours a message or counts towards a sender.
# Even after adding everyday messages to its training data, it flags 43% of our harmless dating messages.
EXPERIMENTAL = {"gaslighting"}

SAMPLE_CHAT = """Sam: hey! loved your hiking photos
Maya: thanks! that one was in the dolomites
Sam: so when are we going for that drink
Maya: can't this week, work is a bit crazy
Sam: you always have an excuse
Sam: you're honestly such a stupid tease
Maya: please don't talk to me like that
Sam: shut up, you're pathetic and nobody else would date you
Maya: i think i'm done here
Sam: you are going to be sorry you ignored me"""

STYLE = """
<style>
.rf{--bg:#fbf7f2;--ink:#1f1b18;--muted:#6b625a;--line:#e6ddd2;--card:#fff;--red:#b3261e;--red-bg:#fbe4e1;--amber:#8a5a00;--amber-bg:#fbeecf;--bubble:#f1ebe3;
 background:var(--bg);color:var(--ink);border-radius:14px;padding:18px;font-family:Georgia,'Times New Roman',serif}
.rf h3{margin:0 0 8px;font-size:16px;color:var(--ink)}
.rf .senders{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}
.rf .sender{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 12px;font:13px/1.4 system-ui,sans-serif;color:var(--ink)}
.rf .sender b{display:block;font-size:14px}
.rf .row-label{font:12px system-ui,sans-serif;color:var(--muted);margin:2px 0 4px}
.rf .timeline{display:flex;align-items:flex-end;gap:3px;height:48px;border-bottom:1px solid var(--line);margin-bottom:6px}
.rf .tick{flex:1;min-width:3px;max-width:24px;border-radius:3px 3px 0 0;background:#c9bfb2}
.rf .tick.flag{background:var(--red)}.rf .tick.watch{background:#d99a1c}
.rf .axis{font:12px system-ui,sans-serif;color:var(--muted);margin-bottom:16px}
.rf .msg{margin:0 0 10px;font:15px/1.45 system-ui,sans-serif}
.rf .who{font-size:12px;color:var(--muted);margin-bottom:2px}
.rf .bubble{display:inline-block;max-width:88%;background:var(--bubble);color:var(--ink);padding:8px 12px;border-radius:12px;overflow-wrap:anywhere}
.rf .msg.flag .bubble{background:var(--red-bg);box-shadow:inset 3px 0 0 var(--red)}
.rf .msg.watch .bubble{background:var(--amber-bg);box-shadow:inset 3px 0 0 #d99a1c}
.rf mark{background:rgba(179,38,30,.2);color:inherit;border-radius:3px;padding:0 2px}
.rf .tag{font-size:12px;margin-top:3px;color:var(--muted)}
.rf .tag .flag{color:var(--red);font-weight:600}.rf .tag .watch{color:var(--amber);font-weight:600}
.rf .exp{font-style:italic}
/* Gradio's dark theme recolours headings and bold text; keep the report readable on its light card. */
.rf h3,.rf .sender,.rf .sender b,.rf .bubble,.rf mark{color:var(--ink)!important}
.rf .tag,.rf .exp,.rf .who,.rf .axis,.rf .row-label,.rf .note{color:var(--muted)!important}
.rf .tag .flag{color:var(--red)!important}.rf .tag .watch{color:var(--amber)!important}
.rf .note{font:13px/1.5 system-ui,sans-serif;color:var(--muted);border-top:1px solid var(--line);padding-top:12px;margin-top:14px}
.rf .empty{font:15px system-ui,sans-serif;color:var(--muted)}
</style>
"""

NOTE = (
    "Support tool only. Only the toxic score raises flags. It catches explicit insults but misses most "
    "put-downs and threats, so a message with no flag can still be harmful. The gaslighting score is "
    "experimental and raises no flags: its training chats come from a therapy setting, and on our dating "
    "messages it still gives high scores to many ordinary messages. Scores are model outputs, not calibrated "
    "chances. If you feel unsafe, trust that feeling and talk to someone you trust or a local support service."
)

CSS_CLASS = {"red flag": "flag", "worth a look": "watch"}


def load_scorer(path: Path) -> tuple[Scorer, float]:
    """Load a saved model and return (scorer, threshold). Handles TF-IDF (.joblib) and CNN (.pt) files."""
    if path.suffix == ".joblib":
        saved = joblib.load(path)
        model = saved["model"]
        return (lambda texts: model.predict_proba(list(texts))[:, 1]), saved["threshold"]

    import torch

    from .cnn import TextCNN, encode, predict_scores

    saved = torch.load(path, map_location="cpu", weights_only=False)
    # The embedding shape comes from the saved weights, so no GloVe file is needed here.
    model = TextCNN(np.zeros(tuple(saved["state_dict"]["embedding.weight"].shape), dtype=np.float32))
    model.load_state_dict(saved["state_dict"])
    vocab, max_len = saved["vocab"], saved["max_len"]
    return (lambda texts: predict_scores(model, encode(list(texts), vocab, max_len), "cpu")), saved["threshold"]


def default_models() -> tuple[dict[str, Path], str]:
    """Use the CNN models when PyTorch is installed, otherwise the TF-IDF baselines."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return BASELINE_MODELS, "TF-IDF baselines (install PyTorch to use the CNN)"
    return CNN_MODELS, "CNN models"


def _tag(topic: str, message: ScoredMessage) -> str:
    if topic in EXPERIMENTAL:
        return f'<span class="exp">{TOPICS[topic]} (experimental, not reliable on dating chats): {message.score:.2f}</span>'
    level = message.level
    css = CSS_CLASS.get(level, "")
    text = f'<span class="{css}">{level.capitalize()}</span>' if css else level
    return f"{TOPICS[topic]}: {text} · {message.score:.2f}"


def _message_html(by_topic: dict[str, ScoredMessage]) -> str:
    first = next(iter(by_topic.values()))
    flagging = {topic: m for topic, m in by_topic.items() if topic not in EXPERIMENTAL}
    worst = max((m.level for m in flagging.values()), key=LEVEL_ORDER.get, default="no flag")
    # Highlight a word if it pushed up a score that raised a flag.
    marked = set()
    for message in flagging.values():
        if message.level != "no flag":
            marked |= {i for i, (_, effect) in enumerate(message.word_effects) if effect >= MIN_WORD_EFFECT}
    words = [f"<mark>{html.escape(w)}</mark>" if i in marked else html.escape(w) for i, w in enumerate(first.text.split())]
    tags = " &nbsp;|&nbsp; ".join(_tag(topic, message) for topic, message in by_topic.items())
    return (
        f'<div class="msg {CSS_CLASS.get(worst, "")}"><div class="who">{html.escape(first.sender)}</div>'
        f'<div class="bubble">{" ".join(words)}</div><div class="tag">{tags}</div></div>'
    )


def _senders_html(reports: dict[str, list[ScoredMessage]]) -> str:
    counted = [t for t in reports if t not in EXPERIMENTAL]
    rows: dict[str, dict] = {}
    for message in next(iter(reports.values())):
        rows.setdefault(message.sender, {"messages": 0, **{t: 0 for t in counted}})["messages"] += 1
    for topic in counted:
        for message in reports[topic]:
            rows[message.sender][topic] += message.level == "red flag"
    ordered = sorted(rows.items(), key=lambda item: (-sum(item[1][t] for t in counted), item[0]))
    return "".join(
        f'<div class="sender"><b>{html.escape(sender)}</b>'
        + "".join(f"{row[t]} {TOPICS[t].lower()} flags · " for t in counted)
        + f'{row["messages"]} messages</div>'
        for sender, row in ordered
    )


def _timeline_html(topic: str, report: list[ScoredMessage]) -> str:
    experimental = topic in EXPERIMENTAL
    ticks = "".join(
        f'<div class="tick {"" if experimental else CSS_CLASS.get(m.level, "")}" style="height:{max(m.score * 100, 4):.0f}%" '
        f'title="Message {i}: score {m.score:.2f}"></div>'
        for i, m in enumerate(report, 1)
    )
    label = f"{TOPICS[topic]} (experimental, grey only)" if experimental else TOPICS[topic]
    return f'<div class="row-label">{label}</div><div class="timeline">{ticks}</div>'


def render_report(reports: dict[str, list[ScoredMessage]] | list[ScoredMessage]) -> str:
    """Build the report as HTML. All chat text passes through html.escape."""
    if isinstance(reports, list):
        reports = {"toxic": reports}
    first = next(iter(reports.values()), [])
    if not first:
        return f'{STYLE}<div class="rf"><p class="empty">Paste a chat to see the report.</p></div>'
    messages = "".join(
        _message_html({topic: report[i] for topic, report in reports.items()}) for i in range(len(first))
    )
    timelines = "".join(_timeline_html(topic, report) for topic, report in reports.items())
    return (
        f'{STYLE}<div class="rf"><h3>Who said what</h3><div class="senders">{_senders_html(reports)}</div>'
        f"<h3>Scores across the chat</h3>{timelines}"
        f'<div class="axis">First message to last. Taller means a higher score.</div>'
        f'<h3>Messages</h3>{messages}<div class="note">{html.escape(NOTE)}</div></div>'
    )


def build_app(model_paths: dict[str, Path] | None = None) -> gr.Blocks:
    if model_paths is None:
        model_paths, model_name = default_models()
    else:
        model_name = "custom models"
    scorers = {topic: load_scorer(path) for topic, path in model_paths.items()}

    def analyse(chat: str) -> str:
        reports = {}
        for topic, (scorer, threshold) in scorers.items():
            # An infinite threshold means an experimental score never reaches "red flag"; render_report ignores its level.
            reports[topic] = score_chat(chat, scorer, float("inf") if topic in EXPERIMENTAL else threshold)
        return render_report(reports)

    with gr.Blocks(title="Red Flag Detector", analytics_enabled=False) as app:
        gr.Markdown(
            "# Red Flag Detector\nPaste a dating chat, one message per line as `Name: message`. "
            "WhatsApp exports work too. We do not save your chat. "
            f"Each message gets a toxic-language score and a gaslighting score from our {model_name}."
        )
        with gr.Row():
            with gr.Column(scale=2):
                chat = gr.Textbox(label="Chat", value=SAMPLE_CHAT, lines=16)
                button = gr.Button("Check this chat", variant="primary")
            with gr.Column(scale=3):
                output = gr.HTML(analyse(SAMPLE_CHAT))
        button.click(analyse, inputs=chat, outputs=output)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--toxic-model", type=Path, help="defaults to models/cnn_d1.pt")
    parser.add_argument("--gaslighting-model", type=Path, help="defaults to models/cnn_d3plus.pt")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    paths = None
    if args.toxic_model or args.gaslighting_model:
        defaults, _ = default_models()
        paths = {"toxic": args.toxic_model or defaults["toxic"], "gaslighting": args.gaslighting_model or defaults["gaslighting"]}
    build_app(paths).launch(server_port=args.port)


if __name__ == "__main__":
    main()
