# Results summary

All numbers come from the single final run on 18 November (commit `4add43c`), in `results/final/`.

## Toxic language (trained on D1, news-site comments)

| Model | D1 test F1 | D2 test F1 | Dating F1 | Dating ROC-AUC |
|---|---:|---:|---:|---:|
| TF-IDF + logistic regression | 0.57 | 0.56 | 0.39 | 0.68 |
| CNN + GloVe (ours) | 0.62 | 0.60 | 0.41 | 0.70 |
| DistilBERT, frozen | 0.43 | 0.55 | 0.55 | 0.88 |
| DistilBERT, fine-tuned | 0.66 | 0.66 | 0.45 | 0.77 |

- On news comments, fine-tuned DistilBERT beats the CNN and the CNN beats the baseline; both gaps are statistically reliable.
- On the 80 final dating messages every model catches all 6 explicit insults but at most 2 of 7 put-downs and 1 of 7 threats. Only frozen DistilBERT beats the baseline reliably.

## Gaslighting (trained on D3, ChatGPT-written therapy chats)

- All eight gaslighting models score F1 0.92 to 0.98 on the D3 test set.
- On dating messages they rank at chance (ROC-AUC 0.33 to 0.58). The D3-only models flag 45 to 68% of harmless dating messages; adding everyday messages cuts that to 3 to 43% and removes most detections.

## Limitations

The dating test sets are small (20 toxic and 7 gaslighting messages, each against 40 harmless), so intervals are wide. We wrote the dating messages ourselves, so real chats may differ. The full analysis is in the report.
