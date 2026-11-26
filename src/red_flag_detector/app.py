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

from .chat import MIN_WORD_EFFECT, ScoredMessage, Scorer, parse_chat, score_chat
from .generator import HARMLESS, generate_chat

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
TOPICS = {"toxic": "Toxic", "gaslighting": "Gaslighting"}
# The gaslighting CNN was trained on D3 plus everyday D1 messages, which cuts its false alarms on dating chat.
CNN_MODELS = {"toxic": MODELS_DIR / "cnn_d1.pt", "gaslighting": MODELS_DIR / "cnn_d3plus.pt"}
BASELINE_MODELS = {"toxic": MODELS_DIR / "baseline_d1_100k.joblib", "gaslighting": MODELS_DIR / "baseline_d3plus.joblib"}
LEVEL_ORDER = {"no flag": 0, "worth a look": 1, "red flag": 2}
# Shown as a grey number only: it never raises a flag, colours a message or counts towards a sender.
# On our final dating test every gaslighting model, this one included, ranked messages at chance level.
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
.rf .truth{font:12px system-ui,sans-serif;margin-top:4px;color:var(--muted)}
.rf .pill{display:inline-block;border-radius:999px;padding:1px 9px;font-weight:600;margin-right:6px}
.rf .pill.caught{background:#dcf1e2;color:#1e6b38!important}.rf .pill.missed{background:var(--red-bg);color:var(--red)!important}
.rf .pill.false{background:var(--amber-bg);color:var(--amber)!important}.rf .pill.ok{background:#ece6de;color:var(--muted)!important}
.rf .score{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 16px}
.rf .score div{flex:1;min-width:120px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 12px;font:13px/1.3 system-ui,sans-serif;color:var(--muted)!important}
.rf .score b{display:block;font-size:26px;color:var(--ink)!important}
.rf .scenario{font:14px system-ui,sans-serif;color:var(--muted)!important;margin:0 0 10px}
.rf .scenario b{color:var(--ink)!important}
</style>
"""

GUIDE_STYLE = """
<style>
.rf .guide h3{margin:16px 0 6px}.rf .guide h3:first-child{margin-top:0}
.rf .guide p,.rf .guide li{font:14px/1.5 system-ui,sans-serif;color:var(--ink)!important;margin:0 0 6px}
.rf .guide ul{margin:0 0 4px;padding-left:18px}
.rf .key{display:flex;align-items:flex-start;gap:10px;margin:0 0 8px;font:14px/1.45 system-ui,sans-serif;color:var(--ink)!important}
.rf .swatch{flex:none;width:14px;height:14px;border-radius:4px;margin-top:3px;background:var(--bubble)}
.rf .swatch.flag{background:var(--red-bg);box-shadow:inset 3px 0 0 var(--red)}
.rf .swatch.watch{background:var(--amber-bg);box-shadow:inset 3px 0 0 #d99a1c}
.rf .guide .safety{border-top:1px solid var(--line);padding-top:10px;margin-top:12px;color:var(--muted)!important}
</style>
"""


def guide_html(threshold: float) -> str:
    """The breakdown shown under the buttons: what the colours mean and where the model goes wrong."""
    return (
        f'{STYLE}{GUIDE_STYLE}<div class="rf guide">'
        "<h3>How to read the result</h3>"
        f'<div class="key"><span class="swatch flag"></span><span><b>Red flag</b>: toxic score of {threshold:.2f} '
        "or more. Only the toxic model raises flags.</span></div>"
        f'<div class="key"><span class="swatch watch"></span><span><b>Worth a look</b>: toxic score between 0.50 and '
        f"{threshold:.2f}.</span></div>"
        '<div class="key"><span class="swatch"></span><span><b>No flag</b>: toxic score under 0.50. '
        "This does not mean the message is safe.</span></div>"
        "<p>On flagged messages, the highlighted words are the ones that pushed the score up.</p>"
        "<h3>What it catches</h3>"
        "<ul><li>Explicit insults and swearing aimed at the other person.</li></ul>"
        "<h3>What it misses</h3>"
        "<ul><li>Polite threats, such as \"i know where you work\".</li>"
        "<li>Put-downs without swear words, such as \"nobody else is going to want you\".</li>"
        "<li>Guilt-tripping and controlling messages.</li></ul>"
        "<h3>False alarms</h3>"
        "<ul><li>Friendly banter with rude words, such as \"loser buys drinks\".</li>"
        "<li>People naming their own identity, such as \"as a gay guy\".</li></ul>"
        "<h3>Gaslighting score</h3>"
        "<p>Experimental, shown in grey and never flags. Its training chats come from a therapy setting, "
        "and on our final dating test it did no better than chance.</p>"
        "<h3>Scores</h3>"
        "<p>Scores are model outputs, not calibrated chances.</p>"
        '<p class="safety">Support tool only. If you feel unsafe, trust that feeling and talk to someone you trust '
        "or a local support service.</p></div>"
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


def _outcome(truth: str, flagged: bool) -> tuple[str, str]:
    """Compare a message's true label with the toxic red flag."""
    if truth == HARMLESS:
        return ("false", "False alarm") if flagged else ("ok", "Correctly left alone")
    if flagged:
        return "caught", "Caught"
    if truth == "gaslighting":
        return "missed", "Missed (the gaslighting score never flags)"
    return "missed", "Missed"


def _flagged(by_topic: dict[str, ScoredMessage]) -> bool:
    return any(m.level == "red flag" for topic, m in by_topic.items() if topic not in EXPERIMENTAL)


def _message_html(by_topic: dict[str, ScoredMessage], truth: str | None = None) -> str:
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
    truth_html = ""
    if truth is not None:
        css, outcome = _outcome(truth, _flagged(by_topic))
        truth_html = f'<div class="truth"><span class="pill {css}">{outcome}</span>Really: {html.escape(truth)}</div>'
    return (
        f'<div class="msg {CSS_CLASS.get(worst, "")}"><div class="who">{html.escape(first.sender)}</div>'
        f'<div class="bubble">{" ".join(words)}</div><div class="tag">{tags}</div>{truth_html}</div>'
    )


def _scoreboard_html(rows: list[dict[str, ScoredMessage]], truths: list[str], title: str | None) -> str:
    flagged = [_flagged(row) for row in rows]
    harmful = [truth != HARMLESS for truth in truths]
    caught = sum(f and h for f, h in zip(flagged, harmful))
    false_alarms = sum(f and not h for f, h in zip(flagged, harmful))
    scenario = f'<p class="scenario">Scenario: <b>{html.escape(title)}</b></p>' if title else ""
    return (
        f'<h3>The truth</h3>{scenario}<div class="score">'
        f"<div><b>{caught} / {sum(harmful)}</b>harmful messages caught</div>"
        f"<div><b>{sum(harmful) - caught}</b>harmful messages missed</div>"
        f"<div><b>{false_alarms}</b>false alarms on {len(truths) - sum(harmful)} harmless messages</div></div>"
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
        + "".join(f"{row[t]} {TOPICS[t].lower()} flag{'' if row[t] == 1 else 's'} · " for t in counted)
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


def render_report(
    reports: dict[str, list[ScoredMessage]] | list[ScoredMessage],
    truths: list[str] | None = None,
    title: str | None = None,
) -> str:
    """Build the report as HTML. All chat text passes through html.escape.

    With truths (one true label per message, from a generated chat), each message also shows whether the
    toxic flag caught it, missed it or raised a false alarm, under a scoreboard.
    """
    if isinstance(reports, list):
        reports = {"toxic": reports}
    first = next(iter(reports.values()), [])
    if not first:
        return f'{STYLE}<div class="rf"><p class="empty">Paste a chat to see the report.</p></div>'
    rows = [{topic: report[i] for topic, report in reports.items()} for i in range(len(first))]
    if truths is not None and len(truths) != len(rows):
        truths = None
    messages = "".join(_message_html(row, truths[i] if truths else None) for i, row in enumerate(rows))
    timelines = "".join(_timeline_html(topic, report) for topic, report in reports.items())
    scoreboard = _scoreboard_html(rows, truths, title) if truths else ""
    return (
        f'{STYLE}<div class="rf">{scoreboard}<h3>Who said what</h3><div class="senders">{_senders_html(reports)}</div>'
        f"<h3>Scores across the chat</h3>{timelines}"
        f'<div class="axis">First message to last. Taller means a higher score.</div>'
        f"<h3>Messages</h3>{messages}</div>"
    )


def build_app(model_paths: dict[str, Path] | None = None) -> gr.Blocks:
    if model_paths is None:
        model_paths, model_name = default_models()
    else:
        model_name = "custom models"
    scorers = {topic: load_scorer(path) for topic, path in model_paths.items()}

    def score_all(chat: str) -> dict[str, list[ScoredMessage]]:
        reports = {}
        for topic, (scorer, threshold) in scorers.items():
            # An infinite threshold means an experimental score never reaches "red flag"; render_report ignores its level.
            reports[topic] = score_chat(chat, scorer, float("inf") if topic in EXPERIMENTAL else threshold)
        return reports

    def analyse(chat: str) -> str:
        return render_report(score_all(chat))

    def new_chat() -> tuple[str, dict, str]:
        generated = generate_chat()
        waiting = (
            f'{STYLE}<div class="rf"><p class="empty">New chat with <b>{len(generated.truths)} messages</b>. '
            "Which ones are red flags? Vote, then press <b>Check this chat</b>.</p></div>"
        )
        return generated.text, {"text": generated.text, "truths": generated.truths, "title": generated.title}, waiting

    def reveal(chat: str, game: dict | None) -> str:
        if not game or chat.strip() != game["text"].strip() or len(parse_chat(chat)) != len(game["truths"]):
            return (
                f'{STYLE}<div class="rf"><p class="empty">The truth is only known for generated chats. '
                "Press <b>Generate a random chat</b> first and leave the chat unedited.</p></div>"
            )
        return render_report(score_all(chat), game["truths"], game["title"])

    with gr.Blocks(title="Red Flag Detector", analytics_enabled=False) as app:
        gr.Markdown(
            "# Red Flag Detector\nPaste a dating chat, one message per line as `Name: message`. "
            "WhatsApp exports work too. We do not save your chat. "
            f"Each message gets a toxic-language score and a gaslighting score from our {model_name}.\n\n"
            "**Red flag or not?** Generate a random chat, let everyone vote on which messages are red flags, "
            "check it, then reveal the truth."
        )
        game = gr.State(None)
        with gr.Row():
            with gr.Column(scale=2):
                chat = gr.Textbox(label="Chat", value=SAMPLE_CHAT, lines=16)
                generate = gr.Button("Generate a random chat")
                button = gr.Button("Check this chat", variant="primary")
                truth = gr.Button("Reveal the truth")
                gr.HTML(guide_html(scorers["toxic"][1]) if "toxic" in scorers else "")
            with gr.Column(scale=3):
                output = gr.HTML(analyse(SAMPLE_CHAT))
        generate.click(new_chat, outputs=[chat, game, output])
        button.click(analyse, inputs=chat, outputs=output)
        truth.click(reveal, inputs=[chat, game], outputs=output)
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
