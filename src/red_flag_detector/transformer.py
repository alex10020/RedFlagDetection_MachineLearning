"""Models 3 and 4: pretrained DistilBERT, frozen and fine-tuned.

DistilBERT: Sanh, Debut, Chaumond and Wolf (2019), arXiv:1910.01108, the
`distilbert/distilbert-base-uncased` checkpoint from Hugging Face. We add our
own output layer on the first token's 768-number vector.

- Frozen: DistilBERT stays fixed. We compute each message's vector once and
  train only the output layer (feature extraction, as in Session 12).
- Fine-tuned: every layer trains, with a smaller learning rate for the
  pretrained part than for the new output layer.

Run with:
    python -m red_flag_detector.transformer --dataset d1 --mode frozen
    python -m red_flag_detector.transformer --dataset d1 --mode finetuned
    (and --dataset d3 for gaslighting)
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .baseline import save_stress_outputs, stress_summary
from .data import D1_TRAIN_SIZE, D1_VAL_SIZE, load_d1_train_val, load_d3_splits, load_stress_test
from .evaluate import best_f1_threshold, binary_metrics
from .validation import validate_no_text_overlap

MODEL_NAME = "distilbert/distilbert-base-uncased"
SEED = 42
MAX_LEN = 128
# Frozen grid: only the output layer trains, so trying a few settings is cheap.
FROZEN_GRID = [{"lr": lr, "weight_decay": decay} for lr in (1e-3, 3e-4) for decay in (0.0, 1e-4)]
# Fine-tuning settings from the BERT and DistilBERT papers' recommended range (2e-5 to 5e-5).
FINETUNE = {"backbone_lr": 2e-5, "head_lr": 1e-4, "weight_decay": 0.01, "epochs": 3, "patience": 1, "batch_size": 32, "warmup": 0.1}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(texts, tokenizer, max_len: int = MAX_LEN) -> list[list[int]]:
    """Turn texts into token ids, cut at `max_len` tokens, without padding yet."""
    return tokenizer(list(texts), truncation=True, max_length=max_len)["input_ids"]


def pad_batch(sequences: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad to the longest sequence in the batch, so short messages cost less."""
    width = max(len(seq) for seq in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, seq in enumerate(sequences):
        ids[row, : len(seq)] = torch.tensor(seq)
        mask[row, : len(seq)] = 1
    return ids, mask


def _length_batches(sequences, batch_size: int):
    """Yield index batches of similar length, which keeps padding small during scoring."""
    order = np.argsort([len(seq) for seq in sequences])
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


class DistilBertClassifier(nn.Module):
    """DistilBERT, then dropout and a linear layer on the first ([CLS]) token."""

    def __init__(self, backbone: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(backbone.config.dim, 2)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        first_token = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.out(self.dropout(first_token))


def class_weights(labels, device) -> torch.Tensor:
    """'Balanced' weights, as in the other models: both classes count equally in the loss."""
    counts = np.bincount(np.asarray(labels), minlength=2)
    return torch.tensor(len(labels) / (2 * counts), dtype=torch.float32, device=device)


@torch.no_grad()
def embed(backbone: nn.Module, sequences, pad_id: int, device: str, batch_size: int = 128) -> torch.Tensor:
    """Return the first-token vector of each message from a frozen backbone."""
    backbone.eval()
    out = torch.empty((len(sequences), backbone.config.dim))
    for index in _length_batches(sequences, batch_size):
        ids, mask = pad_batch([sequences[i] for i in index], pad_id)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            hidden = backbone(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state[:, 0]
        out[torch.as_tensor(index)] = hidden.float().cpu()
    return out


def train_head(
    train_x: torch.Tensor,
    train_y,
    val_x: torch.Tensor,
    val_y,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    dropout: float = 0.1,
    epochs: int = 200,
    patience: int = 5,
    batch_size: int = 256,
    device: str = "cpu",
    seed: int = SEED,
) -> tuple[nn.Module, list[dict]]:
    """Train dropout + linear on frozen vectors with weighted cross-entropy and Adam; stop on validation PR-AUC."""
    _seed_everything(seed)
    head = nn.Sequential(nn.Dropout(dropout), nn.Linear(train_x.shape[1], 2)).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_y, device))
    x, y = train_x.to(device), torch.as_tensor(np.asarray(train_y), dtype=torch.long, device=device)
    generator = torch.Generator().manual_seed(seed)
    history, best, best_state, waited = [], -1.0, None, 0
    for epoch in range(1, epochs + 1):
        head.train()
        for index in torch.randperm(len(x), generator=generator).split(batch_size):
            optimizer.zero_grad()
            loss_fn(head(x[index]), y[index]).backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            scores = torch.softmax(head(val_x.to(device)), dim=1)[:, 1].cpu().numpy()
        metrics = binary_metrics(val_y, scores, best_f1_threshold(val_y, scores))
        history.append({"epoch": epoch, **metrics})
        if metrics["pr_auc"] > best:
            best, waited, best_state = metrics["pr_auc"], 0, {k: v.clone() for k, v in head.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    head.load_state_dict(best_state)
    return head, history


@torch.no_grad()
def predict_scores(model: DistilBertClassifier, sequences, pad_id: int, device: str, batch_size: int = 128) -> np.ndarray:
    """Return the probability of the harmful class for each message."""
    model.eval()
    scores = np.empty(len(sequences), dtype=np.float32)
    for index in _length_batches(sequences, batch_size):
        ids, mask = pad_batch([sequences[i] for i in index], pad_id)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            logits = model(ids.to(device), mask.to(device))
        scores[index] = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
    return scores


def finetune(
    model: DistilBertClassifier,
    train_seqs,
    train_y,
    val_seqs,
    val_y,
    pad_id: int,
    device: str,
    backbone_lr: float = FINETUNE["backbone_lr"],
    head_lr: float = FINETUNE["head_lr"],
    weight_decay: float = FINETUNE["weight_decay"],
    epochs: int = FINETUNE["epochs"],
    patience: int = FINETUNE["patience"],
    batch_size: int = FINETUNE["batch_size"],
    warmup: float = FINETUNE["warmup"],
    seed: int = SEED,
    log=print,
) -> list[dict]:
    """Fine-tune every layer with AdamW, a warm-up then linear decay, mixed precision and early stopping.

    The model keeps the weights of its best epoch. Returns one row of history per epoch.
    """
    from transformers import get_linear_schedule_with_warmup

    _seed_everything(seed)
    model.to(device)
    # Smaller learning rate for the pretrained layers than for the new output layer.
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": list(model.dropout.parameters()) + list(model.out.parameters()), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    steps = epochs * int(np.ceil(len(train_seqs) / batch_size))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(warmup * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_y, device))
    labels = torch.as_tensor(np.asarray(train_y), dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)

    history, best, best_state, waited = [], -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        started, total = time.perf_counter(), 0.0
        for step, index in enumerate(torch.randperm(len(train_seqs), generator=generator).split(batch_size), 1):
            ids, mask = pad_batch([train_seqs[i] for i in index], pad_id)
            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                loss = loss_fn(model(ids.to(device), mask.to(device)).float(), labels[index].to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item() * len(index)
            if step % 500 == 0:
                log(json.dumps({"epoch": epoch, "step": step, "loss": round(total / (step * batch_size), 4)}))
        scores = predict_scores(model, val_seqs, pad_id, device)
        metrics = binary_metrics(val_y, scores, best_f1_threshold(val_y, scores))
        row = {"epoch": epoch, "train_loss": round(total / len(train_seqs), 4), "seconds": round(time.perf_counter() - started, 1), **metrics}
        history.append(row)
        log(json.dumps(row))
        if metrics["pr_auc"] > best:
            best, waited = metrics["pr_auc"], 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    return history


def _load_data(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if dataset == "d1":
        train, val = load_d1_train_val(train_size=D1_TRAIN_SIZE, val_size=D1_VAL_SIZE, seed=SEED)
        return train, val, f"D1 train, shared {len(train)}-message sample"
    train, val, _ = load_d3_splits()
    return train, val, "D3 gaslighting official train split"


def run(dataset: str, mode: str, stress_path: Path, results_dir: Path, model_path: Path) -> dict:
    """Train on D1 (shared sample) or D3, tune on validation, score the dating dev set.

    Official test sets are not scored here: we run each once, at the end, for all models.
    """
    from transformers import AutoModel, AutoTokenizer

    train, val, source = _load_data(dataset)
    validate_no_text_overlap(train, val)
    stress = load_stress_test(stress_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _seed_everything(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    backbone = AutoModel.from_pretrained(MODEL_NAME)
    pad_id = tokenizer.pad_token_id
    train_seqs, val_seqs, stress_seqs = (tokenize(part["text"], tokenizer) for part in (train, val, stress))
    truncated = float(np.mean([len(seq) == MAX_LEN for seq in train_seqs]))

    results_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if mode == "frozen":
        backbone.to(device)
        train_x, val_x, stress_x = (embed(backbone, seqs, pad_id, device) for seqs in (train_seqs, val_seqs, stress_seqs))
        embed_seconds = time.perf_counter() - started
        trials, best = [], None
        for config in FROZEN_GRID:
            head, history = train_head(train_x, train["label"], val_x, val["label"], device=device, **config)
            best_row = max(history, key=lambda row: row["pr_auc"])
            trial = {**config, "best_epoch": best_row["epoch"], "epochs_run": len(history), "val_pr_auc": best_row["pr_auc"], "val_f1": best_row["f1"]}
            trials.append(trial)
            print(json.dumps(trial), flush=True)
            if best is None or trial["val_pr_auc"] > best["trial"]["val_pr_auc"]:
                best = {"head": head, "trial": trial}
        head = best["head"].eval()
        with torch.no_grad():
            val_scores = torch.softmax(head(val_x.to(device)), dim=1)[:, 1].cpu().numpy()
            stress_scores = torch.softmax(head(stress_x.to(device)), dim=1)[:, 1].cpu().numpy()
        # Only the output layer is ours; the backbone comes unchanged from Hugging Face.
        torch.save({"head": head.state_dict(), "backbone": MODEL_NAME, "trained_on": source}, model_path)
        details = {"embed_seconds": round(embed_seconds, 1), "grid": trials, "chosen": best["trial"],
                   "trainable_parameters": sum(p.numel() for p in head.parameters())}
    else:
        model = DistilBertClassifier(backbone)
        history = finetune(model, train_seqs, train["label"], val_seqs, val["label"], pad_id, device)
        val_scores = predict_scores(model, val_seqs, pad_id, device)
        stress_scores = predict_scores(model, stress_seqs, pad_id, device)
        torch.save({"state_dict": model.state_dict(), "backbone": MODEL_NAME, "trained_on": source}, model_path)
        details = {"settings": FINETUNE, "history": history, "best_epoch": max(history, key=lambda row: row["pr_auc"])["epoch"],
                   "trainable_parameters": sum(p.numel() for p in model.parameters())}
    train_seconds = time.perf_counter() - started

    threshold = best_f1_threshold(val["label"], val_scores)
    save_stress_outputs(stress, stress_scores, threshold, results_dir / "dating_by_category.csv", results_dir / "dating_errors.csv")
    report = {
        "dataset": source,
        "mode": mode,
        "backbone": MODEL_NAME,
        "splits": {name: {"rows": len(part), "harmful_share": round(float(part["label"].mean()), 4)} for name, part in (("train", train), ("val", val))},
        "share_truncated_at_128_tokens": round(truncated, 4),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        **details,
        "threshold": threshold,
        "train_seconds": round(train_seconds, 1),
        # Size the app needs: the frozen model also needs the 268 MB backbone from Hugging Face.
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
    parser.add_argument("--mode", choices=["frozen", "finetuned"], default="finetuned")
    parser.add_argument("--stress", type=Path, default=Path("data/dating_stress_test.csv"))
    parser.add_argument("--results", type=Path, help="defaults to results/distilbert_<mode>_<dataset>/")
    parser.add_argument("--model", type=Path, help="defaults to models/distilbert_<mode>_<dataset>.pt")
    args = parser.parse_args()
    name = f"distilbert_{args.mode}_{args.dataset}"
    report = run(args.dataset, args.mode, args.stress, args.results or Path(f"results/{name}"), args.model or Path(f"models/{name}.pt"))
    print(json.dumps({k: v for k, v in report.items() if k not in {"grid", "history"}}, indent=2))


if __name__ == "__main__":
    main()
