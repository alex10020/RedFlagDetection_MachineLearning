# Data

Raw datasets stay out of Git because of their size. Put them in `data/raw/`. All of them contain offensive language.

| ID | Dataset | Use | How to get it |
|---|---|---|---|
| D1 | [SetFit/toxic_conversations](https://huggingface.co/datasets/SetFit/toxic_conversations) | Main training set | `load_toxic_conversations()` downloads it |
| D2 | [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) | Second test, adds toxicity types. Trains the first baseline for now | Download `train.csv` to `data/raw/jigsaw_train.csv`. SHA-256 of our copy: `bd408461...c429d` |
| D3 | [Maxwe11y/gaslighting](https://huggingface.co/datasets/Maxwe11y/gaslighting) | Gaslighting test, split by chat | Loader still to write |
| Ours | `dating_stress_test.csv` | Dating test, in this repo | Already here |

## Dating Stress Test

`dating_stress_test.csv` holds 120 short fictional messages in 12 categories, 10 per category.

| Column | Meaning |
|---|---|
| `label` | 1 harmful, 0 not harmful |
| `category` | The situation the message tests |
| `scope` | `core` matches the proposal (insults, put-downs, threats, plus harmless messages). `stretch` covers pressure, manipulation and gaslighting |
| `label_status` | `draft_ai` until two team members label it independently |

Claude drafted version 0.1. The team still has to review each label, measure agreement between the two labellers, write the remaining 80 messages, and declare the AI draft in the report.

Half the messages are harmful, far more than in real chats, so F1 on this set does not compare directly with F1 on D1 or D2. Compare recall and false positive rate per category instead.

We wrote this version before training any model. Once it is final, keep it frozen: changing messages after seeing errors would leak test information into our choices.
