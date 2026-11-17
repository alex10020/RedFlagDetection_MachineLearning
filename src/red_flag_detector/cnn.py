"""Model 2: a 1D CNN over GloVe word vectors (Kim, 2014).

Each message becomes a sequence of 100-dimensional GloVe vectors. Filters that
span 3, 4 and 5 words slide over it, max pooling keeps the strongest match of
each filter, and dropout plus one linear layer give the score. Unlike TF-IDF,
the filters see word order inside each window.

GloVe: Pennington, Socher and Manning (2014), glove.6B.100d, from the
stanfordnlp/glove mirror on Hugging Face. Put glove.6B.100d.txt in data/raw/.

Run with:
    python -m red_flag_detector.cnn --dataset d1      # shared 100,000-message D1 sample
    python -m red_flag_detector.cnn --dataset d3      # gaslighting
"""

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .baseline import save_stress_outputs, stress_summary
from .data import D1_TRAIN_SIZE, D1_VAL_SIZE, load_d1_train_val, load_d3_splits, load_stress_test
from .evaluate import best_f1_threshold, binary_metrics
from .validation import validate_no_text_overlap

SEED = 42
PAD, UNK = 0, 1
# Words, numbers and single punctuation marks. GloVe has vectors for "?" and "!", which carry tone.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into words and punctuation marks."""
    return TOKEN_PATTERN.findall(text.lower())


def build_vocab(texts, min_count: int = 2, max_size: int = 50_000) -> dict[str, int]:
    """Map the most frequent training words to ids. Index 0 pads, index 1 stands for unknown words."""
    counts = Counter(token for text in texts for token in tokenize(text))
    words = [word for word, count in counts.most_common(max_size) if count >= min_count]
    return {"<pad>": PAD, "<unk>": UNK, **{word: index + 2 for index, word in enumerate(words)}}


def encode(texts, vocab: dict[str, int], max_len: int = 128) -> torch.Tensor:
    """Turn texts into a padded matrix of word ids, cut at `max_len` words."""
    ids = np.full((len(texts), max_len), PAD, dtype=np.int64)
    for row, text in enumerate(texts):
        tokens = [vocab.get(token, UNK) for token in tokenize(text)[:max_len]]
        ids[row, : len(tokens)] = tokens
    return torch.from_numpy(ids)


def load_glove(path: str | Path, vocab: dict[str, int], dim: int = 100, seed: int = SEED) -> tuple[np.ndarray, float]:
    """Build the embedding matrix: GloVe vectors where they exist, small random ones elsewhere.

    Returns the matrix and the share of vocabulary words GloVe covers.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GloVe not found: {path}. See the docstring of red_flag_detector/cnn.py")
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, 0.1, size=(len(vocab), dim)).astype(np.float32)
    matrix[PAD] = 0
    found = 0
    with path.open(encoding="utf8") as file:
        for line in file:
            word, rest = line.split(" ", 1)
            index = vocab.get(word)
            if index is not None and index > UNK:
                matrix[index] = np.asarray(rest.split(), dtype=np.float32)
                found += 1
    return matrix, found / max(len(vocab) - 2, 1)


class TextCNN(nn.Module):
    """Embedding, parallel convolutions over 3, 4 and 5 words, max pooling over time, dropout, linear."""

    def __init__(
        self,
        embeddings: np.ndarray,
        filter_sizes: tuple[int, ...] = (3, 4, 5),
        filters: int = 100,
        dropout: float = 0.5,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.as_tensor(embeddings), freeze=freeze_embeddings, padding_idx=PAD
        )
        dim = embeddings.shape[1]
        self.convs = nn.ModuleList(nn.Conv1d(dim, filters, size) for size in filter_sizes)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(filters * len(filter_sizes), 2)
        self.max_filter = max(filter_sizes)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.shape[1] < self.max_filter:
            # Messages shorter than the widest filter get padded so every filter fits once.
            ids = nn.functional.pad(ids, (0, self.max_filter - ids.shape[1]), value=PAD)
        x = self.embedding(ids).transpose(1, 2)  # (batch, dim, words), the shape Conv1d expects
        pooled = [torch.relu(conv(x)).amax(dim=2) for conv in self.convs]
        return self.out(self.dropout(torch.cat(pooled, dim=1)))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict_scores(model: TextCNN, ids: torch.Tensor, device: str, batch_size: int = 512) -> np.ndarray:
    """Return the probability of the harmful class for each row."""
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch = ids[start : start + batch_size].to(device)
            scores.append(torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(scores) if scores else np.array([])


def train_cnn(
    train: pd.DataFrame,
    val: pd.DataFrame,
    embeddings: np.ndarray,
    vocab: dict[str, int],
    max_len: int = 128,
    epochs: int = 10,
    patience: int = 2,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | None = None,
    seed: int = SEED,
    log=print,
) -> tuple[TextCNN, list[dict]]:
    """Train with class-weighted cross-entropy and Adam; stop when validation PR-AUC stops improving.

    Returns the weights from the best epoch and one row of history per epoch.
    """
    _seed_everything(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TextCNN(embeddings).to(device)
    train_ids, val_ids = encode(train["text"], vocab, max_len), encode(val["text"], vocab, max_len)
    labels = torch.tensor(train["label"].to_numpy(dtype=np.int64))
    # "Balanced" weights, as in the baseline: each class counts as much as the other in the loss.
    counts = np.bincount(train["label"], minlength=2)
    weights = torch.tensor(len(train) / (2 * counts), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)

    history, best_score, best_state, waited = [], -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        started, total = time.perf_counter(), 0.0
        for index in torch.randperm(len(train_ids), generator=generator).split(batch_size):
            optimizer.zero_grad()
            loss = loss_fn(model(train_ids[index].to(device)), labels[index].to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(index)
        val_scores = predict_scores(model, val_ids, device)
        metrics = binary_metrics(val["label"], val_scores, best_f1_threshold(val["label"], val_scores))
        row = {"epoch": epoch, "train_loss": round(total / len(train_ids), 4), "seconds": round(time.perf_counter() - started, 1), **metrics}
        history.append(row)
        log(json.dumps(row))
        if metrics["pr_auc"] > best_score:
            best_score, waited = metrics["pr_auc"], 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


def run(dataset: str, glove_path: Path, stress_path: Path, results_dir: Path, model_path: Path, max_len: int = 128) -> dict:
    """Train on D1 (shared sample) or D3, tune the threshold on validation, score the dating dev set.

    Official test sets are not scored here: we run each once, at the end, for all models.
    """
    if dataset == "d1":
        train, val = load_d1_train_val(train_size=D1_TRAIN_SIZE, val_size=D1_VAL_SIZE, seed=SEED)
        source = f"D1 train, shared {len(train)}-message sample"
    else:
        train, val, _ = load_d3_splits()
        source = "D3 gaslighting official train split"
    validate_no_text_overlap(train, val)

    vocab = build_vocab(train["text"])
    embeddings, coverage = load_glove(glove_path, vocab)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    model, history = train_cnn(train, val, embeddings, vocab, max_len=max_len, device=device)
    train_seconds = time.perf_counter() - started

    val_scores = predict_scores(model, encode(val["text"], vocab, max_len), device)
    threshold = best_f1_threshold(val["label"], val_scores)
    stress = load_stress_test(stress_path)
    stress_scores = predict_scores(model, encode(stress["text"], vocab, max_len), device)

    results_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "vocab": vocab, "threshold": threshold, "max_len": max_len, "trained_on": source},
        model_path,
    )
    save_stress_outputs(stress, stress_scores, threshold, results_dir / "dating_by_category.csv", results_dir / "dating_errors.csv")

    report = {
        "dataset": source,
        "splits": {name: {"rows": len(part), "harmful_share": round(float(part["label"].mean()), 4)} for name, part in (("train", train), ("val", val))},
        "vocab_size": len(vocab),
        "glove_coverage": round(coverage, 4),
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "history": history,
        "best_epoch": max(history, key=lambda row: row["pr_auc"])["epoch"],
        "threshold": threshold,
        "train_seconds": round(train_seconds, 1),
        "model_mb": round(model_path.stat().st_size / 1e6, 2),
        "validation": binary_metrics(val["label"], val_scores, threshold),
        "validation_at_threshold_0.5": binary_metrics(val["label"], val_scores, 0.5),
        "dating_dev_set": stress_summary(stress, stress_scores, threshold),
    }
    (results_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["d1", "d3"], default="d1")
    parser.add_argument("--glove", type=Path, default=Path("data/raw/glove.6B.100d.txt"))
    parser.add_argument("--stress", type=Path, default=Path("data/dating_stress_test.csv"))
    parser.add_argument("--results", type=Path, help="defaults to results/cnn_<dataset>/")
    parser.add_argument("--model", type=Path, help="defaults to models/cnn_<dataset>.pt")
    args = parser.parse_args()
    report = run(
        args.dataset,
        args.glove,
        args.stress,
        args.results or Path(f"results/cnn_{args.dataset}"),
        args.model or Path(f"models/cnn_{args.dataset}.pt"),
    )
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
