"""Collect each model's most confident mistakes for the error analysis.

For every saved model we rescore its validation set and the dating messages,
then keep the highest-scoring false positives and the lowest-scoring false
negatives from validation, plus every dating mistake. Each row gets a draft
cause from simple rules. The `team_cause` column is for the people who read
the messages: the draft rules only sort them, they do not replace reading.

Run with:
    python -m red_flag_detector.errors            # all models
    python -m red_flag_detector.errors --per-side 15

Writes results/errors/<model>.csv and results/errors/summary.csv.
D1 rows quote real online comments, so they contain offensive language.
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import load_d1_train_val, load_d3_splits, load_stress_test

RESULTS = Path("results")
MODELS_DIR = Path("models")
# name -> (kind, topic); the saved file and metrics follow from the name.
MODELS = {
    "d1_baseline_100k": ("baseline", "toxic"),
    "cnn_d1": ("cnn", "toxic"),
    "distilbert_frozen_d1": ("frozen", "toxic"),
    "distilbert_finetuned_d1": ("finetuned", "toxic"),
    "d3_baseline": ("baseline", "gaslighting"),
    "cnn_d3": ("cnn", "gaslighting"),
    "distilbert_frozen_d3": ("frozen", "gaslighting"),
    "distilbert_finetuned_d3": ("finetuned", "gaslighting"),
}
MODEL_FILES = {
    "d1_baseline_100k": "baseline_d1_100k.joblib",
    "d3_baseline": "baseline_d3.joblib",
}

IDENTITY = re.compile(
    r"\b(gays?|lesbians?|trans|muslims?|islam|jews?|jewish|christians?|black|white|women|men|immigrants?|mexicans?)\b", re.I
)
RUDE = re.compile(
    r"\b(f+u+c+k\w*|shit\w*|damn\w*|hell|crap\w*|ass|bitch\w*|stupid\w*|idiot\w*|dumb\w*|moron\w*|pathetic|creep\w*|jerks?"
    r"|clowns?|fools?|losers?|scum|pigs?|pansy|disgusting|ignorant|liars?|trash)\b",
    re.I,
)
FIRST_PERSON = re.compile(r"^(I've|I'm|I |My |Yeah|I just)", re.I)
# Draft causes for dating mistakes come from the category we wrote each message for.
DATING_CAUSES = {
    ("toxic", 1): {
        "threat": "threat without insult words",
        "put_down": "put-down in polite words",
        "explicit_insult": "insult missed",
        "gaslighting": "gaslighting is not toxic language",
        "guilt_manipulation": "guilt or control is not in D1's labels",
        "sexual_pressure": "sexual pressure is not in D1's labels",
    },
    ("toxic", 0): {
        "playful_banter": "friendly teasing with insult words",
        "identity_mention": "identity word",
    },
    ("gaslighting", 1): {"gaslighting": "gaslighting stated plainly, not as a probing question"},
}


def draft_cause(text: str, label: int, topic: str, category: str | None = None) -> str:
    """Return a first-guess cause for one mistake. People confirm or replace it."""
    if category is not None:
        known = DATING_CAUSES.get((topic, label), {}).get(category)
        if known:
            return known
        if label == 0:
            return "harmless chat in a style the training data lacks" if topic == "gaslighting" else "harmless message flagged"
        return f"{category.replace('_', ' ')} is outside the training labels"
    if topic == "toxic":
        if label == 0 and IDENTITY.search(text):
            return "identity word"
        if label == 0 and RUDE.search(text):
            # Often an insult that raters labelled not toxic: check whether the label or the model is wrong.
            return "rude or insulting words, labelled not toxic"
        if label == 1 and not RUDE.search(text):
            return "toxic without rude words"
        return "needs reading"
    if label == 0 and "?" in text:
        return "question in a safe chat"
    if FIRST_PERSON.match(text):
        return "turn reads like the user"
    if label == 1 and "?" not in text:
        return "gaslighting turn without a question"
    return "needs reading"


def pick_errors(data: pd.DataFrame, scores: np.ndarray, threshold: float, per_side: int) -> pd.DataFrame:
    """Keep the `per_side` most confident false positives and false negatives."""
    scored = data.assign(score=scores, flagged=(scores >= threshold).astype(int))
    false_pos = scored[(scored.label == 0) & (scored.flagged == 1)].nlargest(per_side, "score")
    false_neg = scored[(scored.label == 1) & (scored.flagged == 0)].nsmallest(per_side, "score")
    return pd.concat([false_pos.assign(error="false positive"), false_neg.assign(error="false negative")])


def load_scorer(name: str):
    """Return (score_texts, threshold) for a saved model. score_texts maps a list of texts to harmful-class probabilities."""
    kind, _ = MODELS[name]
    threshold = json.loads((RESULTS / name / "metrics.json").read_text())["threshold"]
    if kind == "baseline":
        model = joblib.load(MODELS_DIR / MODEL_FILES[name])["model"]
        return (lambda texts: model.predict_proba(list(texts))[:, 1]), threshold

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved = torch.load(MODELS_DIR / f"{name}.pt", map_location="cpu", weights_only=False)
    if kind == "cnn":
        from .cnn import TextCNN, encode, predict_scores

        model = TextCNN(np.zeros(tuple(saved["state_dict"]["embedding.weight"].shape), dtype=np.float32))
        model.load_state_dict(saved["state_dict"])
        model.to(device)
        return (lambda texts: predict_scores(model, encode(list(texts), saved["vocab"], saved["max_len"]), device)), threshold

    from transformers import AutoModel, AutoTokenizer

    from .transformer import DistilBertClassifier, embed, predict_scores, tokenize

    tokenizer = AutoTokenizer.from_pretrained(saved["backbone"])
    backbone = AutoModel.from_pretrained(saved["backbone"])
    if kind == "frozen":
        head = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(backbone.config.dim, 2))
        head.load_state_dict(saved["head"])
        backbone.to(device)
        head.to(device).eval()

        def score(texts):
            vectors = embed(backbone, tokenize(texts, tokenizer), tokenizer.pad_token_id, device)
            with torch.no_grad():
                return torch.softmax(head(vectors.to(device)), dim=1)[:, 1].cpu().numpy()

        return score, threshold
    model = DistilBertClassifier(backbone)
    model.load_state_dict(saved["state_dict"])
    model.to(device)
    return (lambda texts: predict_scores(model, tokenize(texts, tokenizer), tokenizer.pad_token_id, device)), threshold


def collect(name: str, val: pd.DataFrame, stress: pd.DataFrame, per_side: int) -> pd.DataFrame:
    """Rescore one model and return its validation and dating mistakes with draft causes."""
    _, topic = MODELS[name]
    score, threshold = load_scorer(name)
    val_errors = pick_errors(val[["text", "label"]], score(val["text"]), threshold, per_side).assign(source="validation", category="")
    stress_scores = score(stress["text"])
    flagged = stress_scores >= threshold
    # For a gaslighting model only gaslighting and harmless messages have the right label;
    # missing a threat is not its mistake.
    in_scope = (stress["label"] == 0) | (stress["topic"] == topic) if topic == "gaslighting" else np.ones(len(stress), bool)
    wrong = (flagged != stress["label"].astype(bool)) & in_scope
    dating_errors = stress[wrong][["text", "label", "category"]].assign(
        score=stress_scores[wrong], error=np.where(stress["label"][wrong] == 1, "false negative", "false positive"), source="dating"
    )
    errors = pd.concat([val_errors, dating_errors], ignore_index=True)
    errors["draft_cause"] = [
        draft_cause(row.text, row.label, topic, row.category or None) for row in errors.itertuples()
    ]
    errors["team_cause"] = ""
    errors.insert(0, "model", name)
    return errors[["model", "source", "error", "label", "score", "category", "draft_cause", "team_cause", "text"]].round({"score": 3})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-side", type=int, default=15, help="validation false positives and false negatives kept per model")
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--out", type=Path, default=RESULTS / "errors")
    args = parser.parse_args()

    _, d1_val = load_d1_train_val()
    _, d3_val, _ = load_d3_splits()
    stress = load_stress_test("data/dating_stress_test.csv")
    args.out.mkdir(parents=True, exist_ok=True)
    tables = []
    for name in args.models:
        val = d1_val if MODELS[name][1] == "toxic" else d3_val
        errors = collect(name, val, stress, args.per_side)
        errors.to_csv(args.out / f"{name}.csv", index=False)
        tables.append(errors)
        print(f"{name}: {len(errors)} mistakes", flush=True)
    summary = (
        pd.concat(tables).groupby(["model", "source", "error", "draft_cause"]).size().rename("messages").reset_index()
    )
    summary.to_csv(args.out / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
