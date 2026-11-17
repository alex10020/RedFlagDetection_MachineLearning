import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from red_flag_detector.cnn import PAD, UNK, TextCNN, WindowConv, build_vocab, encode, load_glove, predict_scores, tokenize, train_cnn


def test_tokenize_keeps_question_marks_and_contractions():
    assert tokenize("Are you SURE?? you're imagining it") == ["are", "you", "sure", "?", "?", "you're", "imagining", "it"]


def test_vocab_reserves_pad_and_unknown_and_drops_rare_words():
    vocab = build_vocab(["a b b", "b c"], min_count=2)
    assert vocab == {"<pad>": PAD, "<unk>": UNK, "b": 2}


def test_encode_pads_cuts_and_marks_unknown_words():
    vocab = {"<pad>": 0, "<unk>": 1, "hi": 2}
    ids = encode(["hi there", "hi hi hi hi"], vocab, max_len=3)
    assert ids.tolist() == [[2, 1, 0], [2, 2, 2]]


def test_load_glove_fills_known_words_and_reports_coverage(tmp_path):
    path = tmp_path / "glove.txt"
    path.write_text("hi 1 2\nzebra 3 4\n", encoding="utf8")
    vocab = {"<pad>": 0, "<unk>": 1, "hi": 2, "rare": 3}
    matrix, coverage = load_glove(path, vocab, dim=2)
    assert matrix[2].tolist() == [1, 2]
    assert matrix[PAD].tolist() == [0, 0]
    assert coverage == 0.5


def test_cnn_handles_messages_shorter_than_its_widest_filter():
    model = TextCNN(np.random.default_rng(0).normal(size=(5, 4)).astype(np.float32), filters=3)
    assert model(torch.tensor([[2, 3]])).shape == (1, 2)


def test_window_conv_matches_conv1d():
    conv = torch.nn.Conv1d(4, 3, 2)
    window = WindowConv(4, 3, 2)
    # Conv1d stores weights as (filters, dim, size); unfold flattens each window as dim-major too.
    window.linear.weight.data = conv.weight.data.reshape(3, -1)
    window.linear.bias.data = conv.bias.data
    x = torch.randn(2, 5, 4)
    assert torch.allclose(window(x), conv(x.transpose(1, 2)).transpose(1, 2), atol=1e-6)


def test_cnn_learns_a_separable_toy_task():
    texts = ["you never listen"] * 40 + ["see you soon"] * 40
    data = pd.DataFrame({"text": texts, "label": [1] * 40 + [0] * 40})
    vocab = build_vocab(data["text"], min_count=1)
    embeddings = np.random.default_rng(0).normal(0, 0.1, size=(len(vocab), 8)).astype(np.float32)
    model, history = train_cnn(data, data, embeddings, vocab, max_len=6, epochs=3, device="cpu", log=lambda _: None)
    scores = predict_scores(model, encode(["you never listen", "see you soon"], vocab, 6), "cpu")
    assert scores[0] > scores[1]  # early stopping keeps the best-ranking epoch; the threshold comes later
    assert history[-1]["f1"] == 1
