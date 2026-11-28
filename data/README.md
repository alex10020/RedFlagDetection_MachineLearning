# Data

Raw datasets stay out of Git because of their size. Put them in `data/raw/`. All of them contain offensive language.

| ID | Dataset | Use | How to get it |
|---|---|---|---|
| D1 | [SetFit/toxic_conversations](https://huggingface.co/datasets/SetFit/toxic_conversations) | Main training set | `load_toxic_conversations("train")` or `("test")` downloads the JSON lines file (train 617 MB, test 18 MB) into the Hugging Face cache on first use. `read_toxic_conversations(path)` reads a copy you already have |
| D2 | [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) | Extra test set, with toxicity types | Log in to Kaggle and accept the competition rules. Put `test.csv` and `test_labels.csv` in `data/raw/` and load them with `load_jigsaw_test()`, which drops Kaggle's unscored rows. Do not commit or share the files. Notebook 01 explores `train.csv`, which was our first stand-in training set; to rerun it, put that file at `data/raw/jigsaw_train.csv` |
| D3 | [Maxwe11y/gaslighting](https://huggingface.co/datasets/Maxwe11y/gaslighting) | Main gaslighting training set. Keep its official train, validation and test splits | Download the six JSON files to `data/raw/gaslighting/` (the main README has a one-line command), then use `load_d3_splits()` |
| GloVe | [stanfordnlp/glove](https://huggingface.co/stanfordnlp/glove) | Word vectors for the CNN | Download `glove.6B.zip` (862 MB) and put `glove.6B.100d.txt` in `data/raw/`. Not needed to run the app |
| Ours | `dating_stress_test.csv` | Dating test, in this repo | Already here |

## Dating Stress Test

`dating_stress_test.csv` holds short fictional messages in 12 categories. Messages S001 to S120 (10 per category) are the development set: we have read the models' mistakes on them. Messages S121 to S200 are the final test, written by the team without model output. `load_stress_test()` returns only the development set unless you pass `part="final"`.

| Column | Meaning |
|---|---|
| `label` | 1 harmful, 0 not harmful |
| `category` | The situation the message tests |
| `topic` | `toxic` (insults, put-downs, threats), `gaslighting`, `other_harm` (guilt or control, sexual pressure) or `harmless` |
| `label_status` | `draft_ai` (S001-S120, AI-drafted development set) or `team_reviewed` (S121-S200, written by a team member and reviewed by the group) |

Messages S001 to S120 are a development set. A team member wrote S121 to S200, and the group reviewed them together and agreed with every label. No second person labelled them independently, so we report no inter-annotator agreement. Some messages, such as threats to block or end contact, could reasonably be labelled either way. The report declares the AI draft.

## D3 rules

D3 needs care, and `load_gaslighting()` applies the first two rules:

- Use assistant turns only. The label describes the assistant, and user turns come from the same scenarios in both classes.
- Drop `internal thought`, because it states the speaker's intent and would leak the label.
- Keep the official splits, because safe and gaslighting chats share scenarios.
- ChatGPT wrote this data, and the assistant plays a psychologist, not a date. Our dating test measures how much that gap costs.

## Reading the scores

Half the messages are harmful, far more than in real chats, so F1 on this set does not compare directly with F1 on D1 or D2. Compare recall and false positive rate per category instead.

We wrote this version before training any model. Once it is final, keep it frozen: changing messages after seeing errors would leak test information into our choices.
