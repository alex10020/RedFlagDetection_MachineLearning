# Red Flag Detector

Red Flag Detector looks for two things in dating-app chats: toxic language and gaslighting. Each gets its own score. No public dataset of real dating chats exists, so we train on online comments (toxic) and AI-written chats (gaslighting), then test how well that carries over to dating messages.

**Research question:** can models trained outside dating spot toxic language and gaslighting in dating messages, and does our own CNN do as well as pretrained DistilBERT?

IE University, AI: Machine Learning & Analytics, Group 7: Alexander Glapiak, Alp Kurtbolat, Raya Metchkarova, Milan Josifovikj, Aysel Zeynalova, Sacha Benayoun.

![Prototype v1: each message gets a toxic and a gaslighting score](results/prototype_v1.jpg)

## Status

| Part | State |
|---|---|
| Package structure and tests | Done, 52 tests |
| EDA on D1, D2, D3 and the dating messages | Done (`notebooks/eda.ipynb`) |
| TF-IDF baseline, CNN with GloVe, DistilBERT frozen and fine-tuned | Trained on both topics, plus all four gaslighting models trained again on D3 with everyday messages |
| Error analysis | Done: mistakes exported per model (`results/<model>/dating_errors.csv`) and grouped by cause in the report |
| Dating test | 120 messages (development set) and 80 team-written messages (final test), labels reviewed by the group |
| Final test runs | Done once, 18 November, from commit `4add43c` |
| Prototype | v1 gives a toxic and a gaslighting score from the CNN models. v2 (next step): a model switcher, with fine-tuned DistilBERT hosted as a GitHub release |
| Report and slides | Submitted separately on Blackboard |

## Final results

We scored all 12 models in the comparison once, on held-out test sets, from commit `4add43c` (see `results/final/run_info.json`). Each model kept the threshold it chose on validation data. Full numbers are in `results/final/metrics.csv`, and the paired comparisons in `results/final/paired_differences.csv`. Development-set and validation results stay in each `results/<model>/metrics.json`. The full-data TF-IDF baseline (`results/d1_baseline_all/`) was a validation experiment only and is not in the final run.

**Toxic language** (F1 for the harmful class; the dating subset has 20 toxic and 40 harmless messages)

| Model | D1 test | D2 test | Dating: toxic vs harmless (95% CI) | Dating ROC-AUC | Harmless dating flagged |
|---|---:|---:|---:|---:|---:|
| TF-IDF + logistic regression | 0.57 | 0.56 | 0.39 (0.17 to 0.57) | 0.68 | 23% |
| CNN + GloVe | 0.62 | 0.60 | 0.41 (0.17 to 0.63) | 0.70 | 8% |
| DistilBERT, frozen | 0.43 | 0.55 | 0.55 (0.30 to 0.73) | 0.88 | 10% |
| DistilBERT, fine-tuned | 0.66 | 0.66 | 0.45 (0.20 to 0.65) | 0.77 | 10% |

On D1 and D2, fine-tuned DistilBERT beats the CNN, and the CNN beats the baseline; every one of these gaps has a paired 95% interval above zero. Scores barely drop from D1 (news comments) to D2 (Wikipedia comments). On dating messages only frozen DistilBERT is reliably better than the baseline (+0.16 F1, interval 0.04 to 0.31); the other differences are within noise.

**Gaslighting** (the dating subset has 7 gaslighting and 40 harmless messages)

| Model | D3 test F1 | Dating ROC-AUC | Gaslighting caught | Harmless dating flagged |
|---|---:|---:|---:|---:|
| TF-IDF, D3 only | 0.96 | 0.58 | 4 of 7 | 58% |
| CNN, D3 only | 0.96 | 0.41 | 4 of 7 | 68% |
| DistilBERT frozen, D3 only | 0.94 | 0.49 | 3 of 7 | 45% |
| DistilBERT fine-tuned, D3 only | 0.98 | 0.52 | 5 of 7 | 65% |
| TF-IDF, D3 + everyday messages | 0.95 | 0.43 | 0 of 7 | 13% |
| CNN, D3 + everyday messages | 0.93 | 0.33 | 2 of 7 | 43% |
| DistilBERT frozen, D3 + everyday messages | 0.92 | 0.35 | 0 of 7 | 5% |
| DistilBERT fine-tuned, D3 + everyday messages | 0.97 | 0.36 | 0 of 7 | 3% |

Every gaslighting model scores 0.92 to 0.98 on D3, but on dating messages they rank at chance level (ROC-AUC 0.33 to 0.58, where 0.5 is guessing). D3's gaslighting is a therapist undermining a patient, and its safe class is the same therapist reassuring them, so the models learn the therapy setting's style rather than gaslighting between partners. Adding everyday non-toxic messages to the safe class (12,079 in train, 844 in validation) stops the models flagging ordinary chat, but it does not make them detect gaslighting. On the 120 development messages the fix seemed to help detection; on the final messages it did not, which shows how little a small development set can tell us.

**Research question.** Models trained outside dating catch explicit toxic language in dating messages reasonably well, but miss subtle harm and do not detect gaslighting at all. Our CNN does better than the baseline but not as well as fine-tuned DistilBERT on the training domains; on dating messages none of them is clearly better, except frozen DistilBERT for toxic language. The training data's setting matters more than the model.

## Setup

Use Python 3.12 or newer. We tested on 3.13.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

With conda instead:

```bash
conda env create -f environment.yml
conda activate red-flag-detector
```

PyTorch comes with the base install because the app runs the CNN. Training DistilBERT also needs transformers:

```bash
pip install -e ".[deep]"
```

pip installs the CPU build of PyTorch, which is enough for the app. For training on an NVIDIA GPU, install the CUDA build from pytorch.org first. We trained on an RTX 2070 with `torch` 2.14.1+cu126 and `transformers` 5.23.0.

## Run the prototype

```bash
python -m red_flag_detector.app
```

Open http://127.0.0.1:7860 and paste a chat with one `Name: message` per line. Each message gets two scores, toxic language and gaslighting, from our CNN models in `models/`, so you do not need any data to run it. Highlighted words are the ones that raised a flagged score most.

Only the toxic score raises flags. The gaslighting score appears as a grey number marked experimental and never flags a message. It comes from the CNN trained on D3 plus everyday messages, which ranks our final dating messages at chance level (see Final results). Without PyTorch the app falls back to the TF-IDF baselines. To try other saved models, pass `--toxic-model` or `--gaslighting-model` with a `.pt` or `.joblib` file.

## Get the data

Raw data stays out of Git. All datasets contain offensive language. See [data/README.md](data/README.md) for sources and licences.

| Data | How to get it |
|---|---|
| D1 Toxic Conversations | Nothing to do: the code downloads it from Hugging Face on first use (635 MB) |
| D3 Gaslighting | `python -c "from huggingface_hub import hf_hub_download as d; [d('Maxwe11y/gaslighting', f'{c}_{s}.json', repo_type='dataset', local_dir='data/raw/gaslighting') for c in ('gaslighting', 'anti_gaslighting') for s in ('train', 'validation', 'test')]"` |
| GloVe 6B 100d (CNN only) | Download `glove.6B.zip` (862 MB) from the `stanfordnlp/glove` repo on Hugging Face and put `glove.6B.100d.txt` in `data/raw/` |
| D2 Jigsaw (extra test) | Log in to Kaggle and put `test.csv` and `test_labels.csv` in `data/raw/` |
| Dating messages | Already in `data/dating_stress_test.csv` |

## Reproduce the results

The EDA is in `notebooks/eda.ipynb`: open it with `jupyter notebook notebooks/eda.ipynb` after downloading the data (see Get the data), then run all cells. A short summary of the final results is in `results/summary.md`.

```bash
pytest -q
python -m red_flag_detector.baseline --dataset d1      # TF-IDF, toxic
python -m red_flag_detector.baseline --dataset d3      # TF-IDF, gaslighting
python -m red_flag_detector.baseline --dataset d3plus  # TF-IDF, gaslighting + everyday D1 messages
python -m red_flag_detector.cnn --dataset d1           # CNN, toxic (--dataset d3 or d3plus for gaslighting)
python -m red_flag_detector.transformer --dataset d1 --mode frozen
python -m red_flag_detector.transformer --dataset d1 --mode finetuned
python -m red_flag_detector.errors                     # mistakes per model, into results/errors/
python -m red_flag_detector.final --dry-run            # final pipeline on validation data only
python -m red_flag_detector.final                      # the one-time final run on all test sets
```

Each run writes `results/<model>/metrics.json`, the flag rate per dating category and every dating mistake. On an RTX 2070, the CNN takes about a minute per setting and fine-tuned DistilBERT about 17 minutes on D1.

Fine-tuned DistilBERT weights are 265 MB each, so Git ignores them. Run the fine-tuned commands to recreate them. The other models are in `models/`.

## How we keep the evaluation honest

- Every toxic model trains on the same 100,000 D1 messages and is tuned on the same 20,000. We removed duplicates and every D1 train text that also appears in the official test file.
- D3 keeps its official splits, which share no chats. We use assistant turns only and drop the `internal thought` field, which states the speaker's intent.
- Dating messages S001 to S120 are a development set, because we have read the models' mistakes on them. S121 to S200 are the final test: `load_stress_test()` leaves them out unless you pass `part="final"`.
- Each official test set runs once, at the end.

## The dating messages

Claude drafted messages S001 to S120, which we use as a development set. A team member wrote S121 to S200 following [data/labelling/WRITER_BRIEF.md](data/labelling/WRITER_BRIEF.md), and the group reviewed them together and agreed with every label. We did not have two people label independently, so we report no inter-annotator agreement.
## Project layout

```
├── data/                   dating messages, writer brief, download instructions
├── models/                 saved models (fine-tuned DistilBERT excluded)
├── notebooks/              eda.ipynb: EDA on D1, D2, D3 and the dating messages
├── results/                metrics, figures and mistakes per model
├── src/red_flag_detector/
│   ├── data.py             loaders and splits for D1, D2, D3 and the dating messages
│   ├── validation.py       column, label and leakage checks
│   ├── baseline.py         TF-IDF and logistic regression
│   ├── cnn.py              CNN over GloVe vectors
│   ├── transformer.py      DistilBERT, frozen and fine-tuned
│   ├── evaluate.py         metrics, threshold, intervals
│   ├── errors.py           mistakes per model for error analysis
│   ├── final.py            the one-time final run on all test sets
│   ├── chat.py             chat parsing and word effects
│   ├── plots.py            chart style
│   └── app.py              Gradio prototype
└── tests/
```

## Responsible use

The app supports a person's own judgement and makes no decisions for them. A low score does not mean a message is safe. Pasted chats include another person's messages, so we do not save them.

## AI use

We used Claude (Anthropic) to write the package code, tests, prototype, the EDA notebook and this README, to run the training, and to draft version 0.1 of the dating test. Every team member needs to review and be able to explain this work. The report gives the full declaration.

## References

- Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. EMNLP 2014.
- Li et al. (2025). Can a Large Language Model be a Gaslighter? ICLR 2025. Source of D3.
- Pennington, J., Socher, R. and Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. EMNLP 2014.
- Sanh, V., Debut, L., Chaumond, J. and Wolf, T. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108.
