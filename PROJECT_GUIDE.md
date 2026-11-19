# Project guide

Open this first. It covers what we are building, what we decided and what to do next.

**Group 7:** Alexander Glapiak (contact), Alp Kurtbolat, Raya Metchkarova, Milan Josifovikj, Aysel Zeynalova, Sacha Benayoun  
**Course:** AI: Machine Learning & Analytics, IE University, Prof. Paula Delgado-Santos

| Date | What is due |
|---|---|
| 24 November 2026 | Report, slides, code |
| 30 November and 3 December 2026 | Presentations: 10 minutes plus 2 for questions |

## The project

Red Flag Detector gives each message in a dating chat two scores: toxic language and gaslighting.

**Research question:** can models trained outside dating spot toxic language and gaslighting in dating messages, and does our own CNN do as well as pretrained DistilBERT?

Gaslighting is the hook. It uses ordinary words, so a word-count model barely sees it: "that never happened, you're imagining things again" scores 0.13 on our first baseline, while "you're such an idiot lol i love it" scores 1.00.

## Decisions so far

- Two main topics: toxic language and gaslighting.
- One classifier per topic, because D1 has only toxic labels and D3 has only gaslighting labels.
- Four models per topic: TF-IDF with logistic regression (baseline), a CNN over GloVe vectors, frozen DistilBERT and fine-tuned DistilBERT.
- Every toxic model trains on the same 100,000 D1 messages and is tuned on the same 20,000. D3 keeps its official train, validation and test splits.
- Main metric: F1 for the harmful class. We do not use accuracy.
- Guilt or control messages and sexual pressure stay in the dating test as extra categories.
- The dating test has a development set (S001 to S120, drafted by Claude) and a final test (S121 to S200, written by a team member). The group reviewed the final labels together. We do not double-label or report Cohen's kappa.
- D2 (Kaggle Jigsaw) is an extra test set only. Each person who downloads it accepts the competition rules on their own account, and nobody commits or shares the files.
- Gaslighting has two versions of all four models: trained on D3 alone, and on D3 plus short non-toxic D1 messages as extra safe examples (as many as D3's safe turns, fixed in advance). We report both. On the 120 development messages, the second cut false alarms on harmless messages from 75% to 13% (TF-IDF), 98% to 43% (CNN), 65% to 12% (frozen DistilBERT) and 92% to 3% (fine-tuned DistilBERT), but both DistilBERT models then caught none of the 10 gaslighting messages. On the final test the cut was similar (58% to 13%, 68% to 43%, 45% to 5%, 65% to 3%), and no version detected gaslighting better than chance.
- The final run (`python -m red_flag_detector.final`) scores every model once on all test sets and reports F1, PR-AUC, ROC-AUC and paired bootstrap intervals for the differences between models. A dry run on validation data reproduced every saved validation score.
- The app uses the CNN for both scores, with the D3-plus-everyday CNN for gaslighting. Only the toxic score raises flags. The gaslighting score is a grey number marked experimental, because it still flags 43% of harmless dating messages.
- App v2, with a model switcher and fine-tuned DistilBERT hosted as a GitHub release, is a next step for the report, not for this submission.
- Error analysis reads the CNN and fine-tuned DistilBERT mistakes closely and summarises the other models with counts.
- Dropped: Detoxify comparison, score calibration and an eight-model switch in the app.

Still open: whether Prof. Delgado-Santos accepts a self-written test set of 200 messages if we report results by category.

## Set up VS Code

1. Clone the repo and open its folder in VS Code.
   ```bash
   git clone https://github.com/alex10020/RedFlagDetection_MachineLearning.git
   cd RedFlagDetection_MachineLearning
   ```
2. Install the Python and Jupyter extensions.
3. Create the environment with Python 3.12 or newer.
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```
   This installs PyTorch, so the app runs the CNN. To train DistilBERT, also run `pip install -e ".[deep]"` for transformers. For training on an NVIDIA GPU, install the CUDA build of PyTorch from pytorch.org first.
4. Press `Ctrl+Shift+P`, run **Python: Select Interpreter** and choose `.venv`.
5. Check that everything works.
   ```bash
   pytest -q
   python -m red_flag_detector.app
   ```
   The app opens at http://127.0.0.1:7860.

To retrain the CNN, put `glove.6B.100d.txt` in `data/raw/` (see [data/README.md](data/README.md)).

## Where things are

| Path | What it holds |
|---|---|
| `src/red_flag_detector/data.py` | Loaders for D1, D2, D3 and the dating messages, plus splits |
| `src/red_flag_detector/validation.py` | Column, label and leakage checks |
| `src/red_flag_detector/baseline.py` | TF-IDF and logistic regression |
| `src/red_flag_detector/cnn.py` | CNN over GloVe vectors |
| `src/red_flag_detector/transformer.py` | DistilBERT, frozen and fine-tuned |
| `src/red_flag_detector/evaluate.py` | Metrics, threshold, intervals, category report |
| `src/red_flag_detector/errors.py` | Each model's most confident mistakes |
| `src/red_flag_detector/final.py` | The one-time final run on all test sets |
| `src/red_flag_detector/chat.py` | Chat parsing and word effects |
| `src/red_flag_detector/app.py` | Gradio prototype |
| `notebooks/03_eda_d1_d3.ipynb` | EDA on D1, D3 and the dating messages |
| `data/dating_stress_test.csv` | 200 dating messages: S001-S120 development, S121-S200 final test |
| `results/<model>/` | Metrics, dating flag rate per category and dating mistakes per model |
| `results/errors/` | Mistakes per model for the error analysis, with draft causes |

## Next tasks

Add a name to each task when you take it.

**Done:** all four models on both topics, EDA on D1 and D3, error export, app v1, READMEs, pinned requirements, clean-install check, the 80 final-test messages, the D2 test files and the one-time final run (18 November, commit `4add43c`, results in `results/final/`). The models are frozen: do not change them.

| Task | Due | Owner |
|---|---|---|
| Read the CNN and fine-tuned DistilBERT mistakes in `results/errors/`, fill `team_cause` and pick examples | Causes filled 18 Nov; Milan checks 20 rows by Sat 21 Nov | Milan |
| Business case: check `report/BUSINESS_SOURCES.md`, write report §2 | Sat 21 Nov | Raya |
| Responsible AI: bias, privacy, misuse, limits (report §7) | Sat 21 Nov | Milan |
| Data section (report §3) and slide deck design | Sun 22 Nov | Aysel |
| Prototype section (report §6), screenshots, live demo, PDF assembly | Mon 23 Nov | Sacha |
| Methodology and results (report §4, §5) | Sat 21 Nov | Alp |
| Executive summary, conclusions, final edit, zip and submission | Tue 24 Nov | Alexander |
| Rehearse to 10 minutes (two slides each, see `report/SLIDES_AND_SCRIPT.md`), make sure each of us can explain the code | Mon 23 Nov | Everyone |

## Rules that protect the results

- Choose thresholds and hyperparameters on validation data only.
- The final tests ran once, on 18 November, for all models together: the official D1 and D3 test sets, the D2 test set and dating messages S121 to S200. `load_stress_test()` leaves S121 to S200 out unless you pass `part="final"`.
- Treat the first 120 dating messages as a development set, because we have seen errors on them.
- Use D3's assistant turns only, drop `internal thought` and keep its official splits. See [data/README.md](data/README.md).
- Report intervals on small test sets. Half the dating test is harmful, so its F1 does not compare directly with F1 on comments.
- Keep raw data, Kaggle files, fine-tuned DistilBERT weights, API keys and real private chats out of Git.
- Put reusable code in `src/` and add a test when you add a function.

## Changes from the proposal

Keep this list up to date. The report should explain each change.

| Proposal said | What we did | Why |
|---|---|---|
| One task: toxic or not toxic | Two topics, toxic language and gaslighting, with one classifier each | Gaslighting is the original part of the project. D1 has only toxic labels and D3 only gaslighting labels |
| D3 as a test set without retraining | D3 trains and tunes the gaslighting models; its official test split stays for the final run | A model trained on toxic comments cannot learn gaslighting |
| Gaslighting trained on D3 only | An extra version of all four gaslighting models adds short non-toxic D1 messages to D3's safe class in train and validation | Error analysis showed the D3-only models flag ordinary chat; the fix was judged on the 120 development messages, never on the final 80 |
| 80/20 train/validation split of D1 | A fixed 20,000-message validation set and a 100,000-message training sample from D1 train. Duplicates and the 974 train texts that also appear in D1's test file are removed | Every model sees the same data, and the official test stays unseen |
| The CNN can use more D1 data than DistilBERT | The CNN uses the same 100,000 messages as DistilBERT; only the baseline has an extra full-data run (1.7 million messages) | A fair comparison. The full-data baseline gained 0.06 validation F1 and nothing on dating messages |
| Adam optimiser for the deep models | Adam for the CNN and the frozen DistilBERT head; AdamW with warm-up and linear decay for fine-tuned DistilBERT | AdamW is the standard optimiser for fine-tuning transformers |
| Hyperparameters not specified | Validation grids: CNN (frozen vs fine-tuned GloVe, learning rate, weight decay) and frozen DistilBERT head (learning rate, weight decay) | The rubric asks for justified hyperparameters |
| About 200 messages that we write and label | 120 messages drafted by Claude (development set) and 80 written by a team member (final test), labels reviewed by the group | We saw model errors on the first 120, so the final score needs unseen messages |
| D2 licence: CC0 | CC0, with the comment text under Wikipedia's CC BY-SA 3.0, and Kaggle's rules: academic use allowed, no redistribution | This is what the Kaggle data page and rules say |
| 20+ mistakes for the CNN and DistilBERT | Mistakes exported for all twelve models; the CNN and fine-tuned DistilBERT are read closely | Covers the proposal at a manageable size |
| Streamlit or Gradio app with a red-flag score | Gradio app with a toxic score and an experimental gaslighting score from the CNN | Fine-tuned DistilBERT (265 MB) is too large for GitHub, so a clean install could not run it |
| Metrics: F1, precision, recall | Also PR-AUC, ROC-AUC, bootstrap intervals on the dating test, paired bootstrap intervals for differences between models and flag rate per dating category | Thresholds come from validation sets with a different harmful share, so ROC-AUC shows ranking quality independent of the threshold; paired intervals show whether one model is reliably better |

Implementation note: the CNN writes each convolution as unfold plus a linear layer, which gives the same output as `nn.Conv1d` (a test checks this). cuDNN's Conv1d backward pass ran about 50 times slower on our GPU.

## Limits to discuss in the report

Data:
- D3's safe class is mostly a therapist reassuring someone. Everyday words such as "you", "my" and "me" push the D3 models towards gaslighting, so the D3-only models flag most harmless dating messages (65% to 98% on the development set, 45% to 68% on the final test). Adding everyday messages cuts false alarms to 3% to 43%, but on the final test every gaslighting model, with or without the fix, ranks dating messages at chance level (ROC-AUC 0.33 to 0.58).
- D3 labels describe whole chats, and about 4% of assistant turns read like the user talking, over 80% of them in gaslighting chats.
- ChatGPT wrote D3, and its assistant plays a psychologist, not a date.
- D1 has label noise: some of the models' most confident false positives are insults labelled not toxic.
- In D1, messages that mention identity groups are labelled toxic more often (for example 31% for "gay" against 8% overall).
- The dating messages are written by us, are short (median 9 words), and are half harmful, unlike real chats.
- The dating labels were reviewed by the group but not labelled twice independently, so we report no agreement score. Some messages, such as threats to block or end contact, could reasonably be labelled either way.

Models and evaluation:
- The toxic models learned rude words, not hostility: "I despise you" scores 0.14 while "you are pathetic" scores 1.00.
- On the D1 and D2 tests, fine-tuned DistilBERT and the CNN are reliably better than the baseline (paired intervals above zero). On dating messages only frozen DistilBERT is reliably better, and that rests on 60 messages.
- The final dating subsets are small: 20 toxic and 7 gaslighting messages, each against 40 harmless. Their intervals are wide.
- The everyday-messages fix looked like it helped detection on 70 development messages but not on the final messages: a small development set can mislead.
- Frozen DistilBERT scores below the baseline on D1, because its [CLS] vector was never trained to summarise a message.
- DistilBERT reads at most 128 tokens, which cuts 16% of the D1 training messages. The CNN also cuts at 128 words: in 6 of its 15 most confident D1 validation misses, the only rude words come after the cut (error analysis, 18 November). Dating messages are at most 15 words, so this does not affect them.
- About a third (CNN) to half (fine-tuned DistilBERT) of the 30 most confident D1 validation mistakes are label noise, so D1 F1 slightly understates both models.
- All D3 models score about 0.95 on validation, so the CNN grid on D3 chose between settings within noise.
- Thresholds come from each training set's validation data, not from dating data, and the scores are not calibrated probabilities.
- A single message can hide context. Some gaslighting only shows across a whole chat.
- The fine-tuned DistilBERT weights are not in the repo; a teammate has to retrain them (about 17 minutes on D1).

## Findings log for the report

Findings from checks we ran outside the training scripts, with the numbers to quote. Add to it as you go.

- **Why gaslighting does not transfer.** D3's gaslighting is a ChatGPT-written therapist undermining a patient, and its safe class is the same therapist reassuring them. Gaslighting between dating partners looks different: denying events ("you were drunk, you don't even remember"), rewriting memories, calling the other person dramatic. A model trained on D3 learns the therapy setting's style, so it cannot recognise a partner gaslighting a partner. This is our main conclusion for the gaslighting half of the research question.
- **Final run, toxic language.** On the D1 test, F1 is 0.57 (baseline), 0.62 (CNN), 0.43 (frozen DistilBERT) and 0.66 (fine-tuned DistilBERT); D2 gives almost the same picture (0.56, 0.60, 0.55, 0.66), so the models carry over from news comments to Wikipedia comments. On dating messages (toxic vs harmless), ROC-AUC is 0.68 to 0.88 and only frozen DistilBERT beats the baseline reliably (+0.16 F1). A possible reason: fine-tuning on news comments fits DistilBERT to that setting, while the frozen model keeps more general language knowledge.
- **Final run, gaslighting.** D3 test F1 is 0.92 to 0.98 for all eight gaslighting models, but on dating messages ROC-AUC is 0.33 to 0.58, which is chance level. The D3-only models flag 45% to 68% of harmless dating messages; with everyday messages they flag 3% to 43% but catch 0 to 2 of 7 gaslighting messages.
- **Reproducibility.** A fresh clone, set up only with the README steps (venv, `requirements.txt`, `pip install -e .`), passes all tests and starts the app with the CNN models on CPU PyTorch.

## Business case framing

Our results do not support selling an automatic red-flag detector to daters. Frame the business case around what the evidence shows:

- **Users:** trust and safety teams at dating apps, the people who handle reported messages. Daters are secondary users of the demo app.
- **Value now:** the toxic model as a triage tool that ranks reported messages so human moderators see likely abuse first. It never blocks or bans on its own. That fits our numbers: decent ranking on dating messages (ROC-AUC 0.68 to 0.88), F1 too low for automatic action.
- **Value from the research:** generic toxicity data misses most put-downs, threats and all gaslighting in dating chats. A platform should test any moderation model on its own messages before relying on it, and our evaluation shows how.
- **Business model:** a paid pilot with one dating app. Phase 1: explicit-toxicity triage for moderators. Phase 2: with users' consent, train on messages users themselves reported, the in-domain data our results show is missing, to build gaslighting and coercion detection. Revenue from a per-platform licence or by moderation volume.
- **Responsible AI fit:** a human decides every case, nothing happens automatically, and training data is used only with consent.
- **Still to research, with real sources:** market size, and what features dating apps already have against harassment. Do not quote numbers or competitor features without a source.

## Marking scheme

| Criterion | Points | Where we stand |
|---|---:|---|
| Problem, impact, business value | 1.5 | Problem clear. Users, business model and responsible AI to write |
| Originality | 0.5 | Gaslighting classifier (the D3 paper trains none) and our own dating test |
| Data | 1.5 | D1, D3 and dating EDA done; cleaning and leakage checks documented |
| Methodology and models | 2.0 | Four models on both topics, tuned on validation |
| Evaluation and results | 1.5 | Final tests done with paired comparisons. Error reading to do |
| Prototype and code | 1.5 | App v1 runs from a clean install; 52 tests |
| Report | 0.5 | Not started |
| Presentation | 1.0 | Not started |

The report must also include a contribution table and a declaration of AI use. The professor can adjust an individual grade if the contribution table or Q&A shows someone did not contribute, so keep commits under your own account and learn the code you present.

## Git

Branch from `main`, commit small steps under your own account, then open a pull request.

```bash
git checkout main && git pull
git checkout -b your-task-name
```

## AI use

Log any AI help as you go, so the declaration is easy to write.

- Claude drafted the package code, tests, prototype v0, this guide and dating messages S001 to S120.
- Claude (Claude Code, 17 to 18 November) trained all twelve models, including the four gaslighting models on D3 plus everyday messages, and ran the CNN and frozen DistilBERT grids. It wrote `cnn.py`, `transformer.py`, `errors.py`, `final.py`, the D3-plus-everyday data step, the EDA notebook 03, app v1, the READMEs, the writer brief and the draft error causes, and drafted pull request descriptions.
- Claude ran the leakage checks, the one-time final run (from commit `4add43c`) and the cleanup of outdated v0 files, and drafted the business case framing in this guide. Alp decided which changes to make and reviewed the results.
- Alp wrote dating messages S121 to S200 without AI, and the group reviewed their labels.
