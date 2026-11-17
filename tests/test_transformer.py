import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from red_flag_detector.transformer import DistilBertClassifier, embed, finetune, pad_batch, predict_scores, train_head


def _tiny_backbone():
    """A small random DistilBERT, so tests run offline and fast."""
    config = transformers.DistilBertConfig(vocab_size=50, dim=16, hidden_dim=32, n_layers=1, n_heads=2, max_position_embeddings=32)
    return transformers.DistilBertModel(config)


def test_pad_batch_pads_to_longest_and_masks_padding():
    ids, mask = pad_batch([[5, 6, 7], [8]], pad_id=0)
    assert ids.tolist() == [[5, 6, 7], [8, 0, 0]]
    assert mask.tolist() == [[1, 1, 1], [1, 0, 0]]


def test_classifier_returns_two_logits_per_message():
    model = DistilBertClassifier(_tiny_backbone())
    ids, mask = pad_batch([[1, 2, 3], [4]], pad_id=0)
    assert model(ids, mask).shape == (2, 2)


def test_embed_keeps_message_order_despite_length_sorting():
    backbone = _tiny_backbone()
    seqs = [[1, 2, 3, 4], [5], [6, 7]]
    together = embed(backbone, seqs, pad_id=0, device="cpu", batch_size=2)
    alone = embed(backbone, [seqs[1]], pad_id=0, device="cpu")
    assert torch.allclose(together[1], alone[0], atol=1e-5)


def test_frozen_head_learns_separable_vectors():
    x = torch.cat([torch.randn(40, 8) + 3, torch.randn(40, 8) - 3])
    y = np.array([1] * 40 + [0] * 40)
    head, history = train_head(x, y, x, y, lr=1e-2, epochs=20, patience=20, batch_size=16)
    assert history[-1]["f1"] == 1


def test_finetune_learns_a_toy_task_and_scores_in_order():
    seqs = [[1, 7, 8]] * 32 + [[1, 9, 10]] * 32
    y = np.array([1] * 32 + [0] * 32)
    model = DistilBertClassifier(_tiny_backbone())
    history = finetune(model, seqs, y, seqs, y, pad_id=0, device="cpu", head_lr=1e-2, backbone_lr=1e-3,
                       epochs=3, patience=3, batch_size=16, log=lambda _: None)
    scores = predict_scores(model, [[1, 7, 8], [1, 9, 10]], pad_id=0, device="cpu")
    assert scores[0] > scores[1]
    assert history[-1]["pr_auc"] == 1
