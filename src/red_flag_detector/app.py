"""Prototype v0. Run with: python -m red_flag_detector.app"""

import argparse
import html
from pathlib import Path

import gradio as gr
import joblib

from .chat import MIN_WORD_EFFECT, ScoredMessage, score_chat, summarise_senders

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "baseline_v0.joblib"

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
.rf .timeline{display:flex;align-items:flex-end;gap:3px;height:64px;border-bottom:1px solid var(--line);margin-bottom:6px}
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
.rf .msg.flag .tag b{color:var(--red)}.rf .msg.watch .tag b{color:var(--amber)}
.rf .note{font:13px/1.5 system-ui,sans-serif;color:var(--muted);border-top:1px solid var(--line);padding-top:12px;margin-top:14px}
.rf .empty{font:15px system-ui,sans-serif;color:var(--muted)}
</style>
"""

NOTE = (
    "Support tool only. Baseline v0 reacts to rude words and misses most put-downs and threats, "
    "so a message with no flag can still be harmful. Scores are model outputs, not calibrated chances. "
    "If you feel unsafe, trust that feeling and talk to someone you trust or a local support service."
)


def _message_html(message: ScoredMessage) -> str:
    words = []
    for word, effect in message.word_effects:
        safe = html.escape(word)
        words.append(f"<mark>{safe}</mark>" if effect >= MIN_WORD_EFFECT else safe)
    css = {"red flag": "flag", "worth a look": "watch"}.get(message.level, "")
    tag = f"<b>{message.level.capitalize()}</b> · score {message.score:.2f}" if css else f"score {message.score:.2f}"
    return (
        f'<div class="msg {css}"><div class="who">{html.escape(message.sender)}</div>'
        f'<div class="bubble">{" ".join(words)}</div><div class="tag">{tag}</div></div>'
    )


def render_report(report: list[ScoredMessage]) -> str:
    """Build the report as HTML. All chat text passes through html.escape."""
    if not report:
        return f'{STYLE}<div class="rf"><p class="empty">Paste a chat to see the report.</p></div>'
    senders = "".join(
        f'<div class="sender"><b>{html.escape(row["sender"])}</b>'
        f'{row["red_flags"]} red flags · {row["worth_a_look"]} worth a look · {row["messages"]} messages</div>'
        for row in summarise_senders(report)
    )
    css = {"red flag": "flag", "worth a look": "watch"}
    ticks = "".join(
        f'<div class="tick {css.get(m.level, "")}" style="height:{max(m.score * 100, 4):.0f}%" '
        f'title="Message {i}: {m.level}, score {m.score:.2f}"></div>'
        for i, m in enumerate(report, 1)
    )
    messages = "".join(_message_html(message) for message in report)
    return (
        f'{STYLE}<div class="rf"><h3>Who said what</h3><div class="senders">{senders}</div>'
        f'<h3>Score across the chat</h3><div class="timeline">{ticks}</div>'
        f'<div class="axis">First message to last. Taller means a higher score.</div>'
        f'<h3>Messages</h3>{messages}<div class="note">{html.escape(NOTE)}</div></div>'
    )


def build_app(model_path: Path = DEFAULT_MODEL) -> gr.Blocks:
    saved = joblib.load(model_path)
    model, threshold = saved["model"], saved["threshold"]

    def scorer(texts):
        return model.predict_proba(list(texts))[:, 1]

    def analyse(chat: str) -> str:
        return render_report(score_chat(chat, scorer, threshold))

    with gr.Blocks(title="Red Flag Detector", analytics_enabled=False) as app:
        gr.Markdown(
            "# Red Flag Detector\nPaste a dating chat, one message per line as `Name: message`. "
            "WhatsApp exports work too. We do not save your chat."
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app(args.model).launch(server_port=args.port)


if __name__ == "__main__":
    main()
