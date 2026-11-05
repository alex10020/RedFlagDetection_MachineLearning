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

Gaslighting is the hook. It uses ordinary words, so a word-count model barely sees it: "that never happened, you're imagining things again" scores 0.13 on our baseline, while "you're such an idiot lol i love it" scores 1.00.

## Decisions so far

- Two main topics: toxic language and gaslighting.
- One classifier per topic, because D1 has only toxic labels and D3 has only gaslighting labels.
- Guilt or control messages and sexual pressure stay in the dating test as extra categories.
- GloVe and DistilBERT stay in the plan.
- Main metric: F1 for the harmful class. We do not use accuracy.

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
4. Press `Ctrl+Shift+P`, run **Python: Select Interpreter** and choose `.venv`.
5. Check that everything works.
   ```bash
   pytest -q
   python -m red_flag_detector.app
   ```
   The app opens at http://127.0.0.1:7860.

For the deep models, add `pip install -e ".[deep]"`. Use Colab or one of our GPUs for training, then commit the code and results back here.

## Where things are

| Path | What it holds |
|---|---|
| `src/red_flag_detector/data.py` | Loaders for D1, D2, D3 and our test, plus splits |
| `src/red_flag_detector/validation.py` | Column, label and leakage checks |
| `src/red_flag_detector/baseline.py` | TF-IDF and logistic regression |
| `src/red_flag_detector/evaluate.py` | Metrics, threshold, intervals, category report |
| `src/red_flag_detector/chat.py` | Chat parsing and word effects |
| `src/red_flag_detector/app.py` | Gradio prototype |
| `data/dating_stress_test.csv` | Our 120 draft test messages |
| `results/summary.md` | Current results and their limits |

## Next tasks

Add a name to each task when you take it.

### By 10 November

| Task | Owner |
|---|---|
| Agree the work split and start the contribution table | |
| Run the baseline and EDA on D1 in Colab | |
| Download D3, run `load_gaslighting()` on the real files and add its EDA | |
| Train the gaslighting baseline on D3 | |
| Review the 120 draft messages: two people label them separately, then measure agreement | |
| Write 80 new messages and do not look at model output on them until the end | |

### By 17 November

| Task | Owner |
|---|---|
| Build the CNN with GloVe in `src/red_flag_detector/cnn.py`: 1D convolutions over 3 to 5 words, max pooling, dropout | |
| Build DistilBERT, frozen and fine-tuned, in `src/red_flag_detector/transformer.py` | |
| Train each model on each topic, with weighted CrossEntropyLoss, Adam and early stopping | |
| Fill the comparison table: F1, drop on dating messages, training time, model size | |
| Read at least 20 mistakes per main model and group them by cause | |

### By 24 November

| Task | Owner |
|---|---|
| Calibrate the scores on validation data, because class weights distort them | |
| Check bias with messages that mention identity | |
| Add the gaslighting score to the prototype and let the app switch models | |
| Write the business case: users, value, business model, market | |
| Write the report (10 to 15 pages) and slides, and rehearse to 10 minutes | |
| Run a clean install from the README and make sure each of us can explain the code | |

## Rules that protect the results

- Choose thresholds and hyperparameters on validation data only.
- Run each official test set once, at the end.
- Treat the first 120 dating messages as a development set, because we have seen errors on them. Use the 80 new messages for the final score.
- Use D3's assistant turns only, drop `internal thought` and keep its official splits. See [data/README.md](data/README.md).
- Report intervals on small test sets. Half the dating test is harmful, so its F1 does not compare directly with F1 on comments.
- Keep raw data, API keys and real private chats out of Git.
- Put reusable code in `src/` and add a test when you add a function.

## Limits to raise ourselves

- We trained the first baseline on D2 as a stand-in. D1 is still the main toxic dataset.
- ChatGPT wrote D3, and its assistant plays a psychologist, not a date.
- We wrote the dating test ourselves, so real chats may look different.
- A single message can hide context. Some gaslighting only shows across a whole chat.

## Marking scheme

| Criterion | Points | Where we stand |
|---|---:|---|
| Problem, impact, business value | 1.5 | Problem clear. Users and business model missing |
| Originality | 0.5 | Gaslighting and our dating test help here |
| Data | 1.5 | D2 and dating EDA done. D1 and D3 missing |
| Methodology and models | 2.0 | Baseline done. CNN and DistilBERT missing |
| Evaluation and results | 1.5 | Pipeline done. Model comparison missing |
| Prototype and code | 1.5 | v0 runs. Gaslighting score missing |
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

Claude drafted the package code, tests, prototype, this guide and version 0.1 of the dating test. Log any further AI help as you go, so the declaration is easy to write.
