"""The final, one-time evaluation of every model on the held-out test sets.

Toxic models are scored on the official D1 test set, the D2 Jigsaw test set and
the final dating messages (S121-S200). Gaslighting models are scored on the
official D3 test set and the final dating messages. Every model keeps the
threshold it chose on validation data; nothing is tuned here.

Run once, after all models are frozen:
    python -m red_flag_detector.final

To check the pipeline without touching any test set, use the validation and
development data instead:
    python -m red_flag_detector.final --dry-run

Writes results/final/ (or results/final_dry_run/):
    metrics.csv              one row per model and test set
    paired_differences.csv   model A minus model B, with 95% paired bootstrap intervals
    dating_by_category.csv   flag rate per dating category and model
    run_info.json            when and from which commit the run was made
Raw scores go to <out>/scores/, which Git ignores because D2's rules forbid
redistributing the data.
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import (
    load_d1_train_val,
    load_d3_splits,
    load_jigsaw_test,
    load_stress_test,
    load_toxic_conversations,
)
from .errors import MODELS, load_scorer
from .evaluate import binary_metrics, bootstrap_interval, category_report, paired_bootstrap_difference

STRESS = Path("data/dating_stress_test.csv")
# Comparisons for the report: every model against its topic's baseline, the two deep
# models we built ourselves against each other, and each D3+everyday model against
# the same model trained on D3 alone.
PAIRS = [
    ("cnn_d1", "d1_baseline_100k"),
    ("distilbert_frozen_d1", "d1_baseline_100k"),
    ("distilbert_finetuned_d1", "d1_baseline_100k"),
    ("distilbert_finetuned_d1", "cnn_d1"),
    ("cnn_d3", "d3_baseline"),
    ("distilbert_frozen_d3", "d3_baseline"),
    ("distilbert_finetuned_d3", "d3_baseline"),
    ("distilbert_finetuned_d3", "cnn_d3"),
    ("d3plus_baseline", "d3_baseline"),
    ("cnn_d3plus", "cnn_d3"),
    ("distilbert_frozen_d3plus", "distilbert_frozen_d3"),
    ("distilbert_finetuned_d3plus", "distilbert_finetuned_d3"),
    ("distilbert_finetuned_d3plus", "cnn_d3plus"),
]


def test_sets(dry_run: bool) -> dict[str, dict[str, pd.DataFrame]]:
    """Return {topic: {test name: data}}. A dry run uses validation and development data only."""
    if dry_run:
        _, d1 = load_d1_train_val()
        _, d3, _ = load_d3_splits()
        dating = load_stress_test(STRESS, part="dev")
        toxic = {"D1 validation": d1}
        gaslighting = {"D3 validation": d3}
    else:
        dating = load_stress_test(STRESS, part="final")
        toxic = {
            "D1 test": load_toxic_conversations("test"),
            "D2 test": load_jigsaw_test("data/raw/test.csv", "data/raw/test_labels.csv"),
        }
        _, _, d3_test = load_d3_splits()
        gaslighting = {"D3 test": d3_test}
    harmless = dating["label"] == 0
    toxic["Dating: toxic vs harmless"] = dating[(dating["topic"] == "toxic") | harmless]
    toxic["Dating: all harmful vs harmless"] = dating
    gaslighting["Dating: gaslighting vs harmless"] = dating[(dating["topic"] == "gaslighting") | harmless]
    return {"toxic": toxic, "gaslighting": gaslighting}


def score_metrics(y, scores, threshold: float, small: bool) -> dict:
    """Threshold metrics, PR-AUC and ROC-AUC; an F1 interval and false-positive rate for small sets."""
    y = np.asarray(y)
    row = {"messages": len(y), "harmful": int(y.sum()), **binary_metrics(y, scores, threshold)}
    row["roc_auc"] = float(roc_auc_score(y, scores)) if 0 < y.sum() < len(y) else float("nan")
    flags = (np.asarray(scores) >= threshold).astype(int)
    row["false_positive_rate"] = float(flags[y == 0].mean()) if (y == 0).any() else float("nan")
    if small:
        row["f1_ci_low"], row["f1_ci_high"] = bootstrap_interval(y, flags)
    return row


def run(dry_run: bool, out: Path, n_resamples: int) -> None:
    started = time.perf_counter()
    sets = test_sets(dry_run)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    rows, scored, categories = [], {}, []
    for name, (_, topic) in MODELS.items():
        score, threshold = load_scorer(name)
        for test_name, data in sets[topic].items():
            scores = np.asarray(score(data["text"]))
            scored[(name, test_name)] = (scores, threshold)
            np.save(out / "scores" / f"{name}__{test_name.replace(' ', '_').replace(':', '')}.npy", scores)
            small = len(data) < 1000
            rows.append({"model": name, "topic": topic, "test_set": test_name, "threshold": threshold,
                         **score_metrics(data["label"], scores, threshold, small)})
            if test_name == "Dating: all harmful vs harmless" or test_name == "Dating: gaslighting vs harmless":
                categories.append(category_report(data, scores, threshold).assign(model=name))
        print(f"{name}: scored", flush=True)

    pairs = []
    for a, b in PAIRS:
        topic = MODELS[a][1]
        for test_name, data in sets[topic].items():
            if (a, test_name) not in scored or (b, test_name) not in scored:
                continue
            (sa, ta), (sb, tb) = scored[(a, test_name)], scored[(b, test_name)]
            for metric in ("f1", "roc_auc"):
                diff = paired_bootstrap_difference(data["label"], sa, ta, sb, tb, metric=metric, n_resamples=n_resamples)
                pairs.append({"model_a": a, "model_b": b, "test_set": test_name, "metric": metric, **diff,
                              "a_reliably_better": diff["ci_low"] > 0, "b_reliably_better": diff["ci_high"] < 0})
        print(f"compared {a} vs {b}", flush=True)

    pd.DataFrame(rows).round(4).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(pairs).round(4).to_csv(out / "paired_differences.csv", index=False)
    pd.concat(categories).round(3).to_csv(out / "dating_by_category.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (out / "run_info.json").write_text(json.dumps({
        "dry_run": dry_run,
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "models": list(MODELS),
        "test_sets": {topic: {name: len(data) for name, data in tests.items()} for topic, tests in sets.items()},
        "bootstrap_resamples": n_resamples,
        "seconds": round(time.perf_counter() - started, 1),
    }, indent=2))
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="use validation and development data, never the test sets")
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args()
    out = Path("results/final_dry_run" if args.dry_run else "results/final")
    if not args.dry_run and (out / "metrics.csv").exists():
        raise SystemExit(f"{out / 'metrics.csv'} already exists. The final run happens once; delete it yourself only if you are sure.")
    run(args.dry_run, out, args.resamples)


if __name__ == "__main__":
    main()
