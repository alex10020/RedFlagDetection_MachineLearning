# Data

Raw datasets stay out of Git because of their size. Put them in `data/raw/`. All of them contain offensive language.

| ID | Dataset | Use | How to get it |
|---|---|---|---|
| D1 | [SetFit/toxic_conversations](https://huggingface.co/datasets/SetFit/toxic_conversations) | Main training set | `load_toxic_conversations()` downloads it |
| D2 | [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) | Second test, adds toxicity types. Trains the first baseline for now | Download `train.csv` to `data/raw/jigsaw_train.csv`. SHA-256 of our copy: `bd408461...c429d` |
| D3 | [Maxwe11y/gaslighting](https://huggingface.co/datasets/Maxwe11y/gaslighting) | Main gaslighting training set. Keep its official train, validation and test splits | Download the JSON files to `data/raw/gaslighting/`, then use `load_gaslighting()` |
| Ours | `dating_stress_test.csv` | Dating test, in this repo | Already here |

## Dating Stress Test

`dating_stress_test.csv` holds 120 short fictional messages in 12 categories, 10 per category.

| Column | Meaning |
|---|---|
| `label` | 1 harmful, 0 not harmful |
| `category` | The situation the message tests |
| `topic` | `toxic` (insults, put-downs, threats), `gaslighting`, `other_harm` (guilt or control, sexual pressure) or `harmless` |
| `label_status` | `draft_ai` until two team members label it independently |

Claude drafted version 0.1. The team still has to review each label, measure agreement between the two labellers, write the remaining 80 messages, and declare the AI draft in the report.

## D3 rules

D3 needs care, and `load_gaslighting()` applies the first two rules:

- Use assistant turns only. The label describes the assistant, and user turns come from the same scenarios in both classes.
- Drop `internal thought`, because it states the speaker's intent and would leak the label.
- Keep the official splits, because safe and gaslighting chats share scenarios.
- ChatGPT wrote this data, and the assistant plays a psychologist, not a date. Our dating test measures how much that gap costs.

## Reading the scores

Half the messages are harmful, far more than in real chats, so F1 on this set does not compare directly with F1 on D1 or D2. Compare recall and false positive rate per category instead.

We wrote this version before training any model. Once it is final, keep it frozen: changing messages after seeing errors would leak test information into our choices.
