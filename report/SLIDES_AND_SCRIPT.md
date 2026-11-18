# Slides and 10-minute script (draft)

> **DRAFT (written by Claude, 18 November).** 12 slides, about 1,250 spoken words, which is roughly 10 minutes at a calm pace. Each speaker should reword their lines so they sound like themselves, then time a full run. If it runs long, cut slide 7 to one sentence first. Six speakers with two slides each, matching the contribution table in the report. Swap slides if you like, but keep two each.
>
> Visual style: reuse the 5 November check-in deck (dark title slide, cream content slides, red for flags), so the two decks look like one project.

| # | Slide | Speaker | Time |
|---|---|---|---:|
| 1 | Title and hook | Alexander | 0:40 |
| 2 | The problem and who has it | Raya | 0:50 |
| 3 | Research question and plan | Alexander | 0:40 |
| 4 | Data: three sources and our own test | Aysel | 1:00 |
| 5 | What the data told us (EDA) | Aysel | 0:50 |
| 6 | Four models | Alp | 1:00 |
| 7 | Keeping the test honest | Milan | 0:40 |
| 8 | Result 1: toxic language | Alp | 1:00 |
| 9 | Result 2: gaslighting | Milan | 1:00 |
| 10 | Demo / prototype | Sacha | 0:50 |
| 11 | Business case and responsible AI | Raya | 1:00 |
| 12 | Conclusion and next steps | Sacha | 0:30 |
| | **Total** | | **10:00** |

---

## Slide 1: Title and hook

**On the slide:** "Red Flag Detector". The research question in one line. Two chat bubbles:
- "you're such an idiot lol i love it"
- "that never happened, you're imagining things again"

Below them: "Which one is the red flag?" Team names, Group 7.

**Script:** "Here are two messages from a dating chat. The first has an insult in it, but it's a joke. The second has no rude word at all, but it's gaslighting: making someone doubt their own memory. A typical toxicity filter flags the first and misses the second. We wanted to know if machine learning can do better, using only data from outside dating, because no public dataset of real dating chats exists."

## Slide 2: The problem and who has it

**On the slide:** Harm in dating chats ranges from explicit insults to put-downs, threats, guilt-tripping, sexual pressure and gaslighting. Who handles it: users report, and trust & safety moderators read the reports. Two numbers: "48% of US online daters faced harassment (Pew 2023)" and "53% of women on dating apps experienced gaslighting (BIT/Minderoo 2026)". (Both checked against the source pages.)

**Script:** "Dating apps depend on users reporting abuse, and then a human moderator has to read every report. The harmful messages aren't all obvious. Some are insults, but many are quiet: a put-down dressed as a joke, a threat in polite words, or gaslighting. Almost half of US online daters have been harassed, and in a 2026 survey of women on dating apps, more than half had been gaslit, a harm that's reported far less than insults. So our users aren't daters first. They're the trust and safety teams who need to know which reports to read first."

## Slide 3: Research question and plan

**On the slide:** "Can models trained outside dating spot toxic language and gaslighting in dating messages, and does our own CNN do as well as pretrained DistilBERT?" Two topics, one classifier each, four models each, one final test on dating messages.

**Script:** "Our question has two halves. First, does a model trained on other text transfer to dating? Second, does the model choice matter: our own CNN against a pretrained transformer? We split harm into two topics, toxic language and gaslighting, trained one classifier for each, and compared four models per topic."

## Slide 4: Data

**On the slide:** a three-column graphic.
- D1, toxic: 1.75 M news comments, 8% toxic → 100 k train / 20 k validation / 50 k official test
- D3, gaslighting: 2,000 AI-written chats, 50% gaslighting, official splits
- Our dating test: 200 messages in 12 categories → 120 development / 80 final, written blind

Footnote: D2, Wikipedia comments, used as an extra test only.

**Script:** "For toxic language we used 1.75 million news-site comments, of which only 8% are toxic. That imbalance is why we report F1 and never accuracy. For gaslighting the only labelled dataset is 2,000 conversations written by ChatGPT, from an ICLR 2025 paper. And because no dating data exists, we wrote 200 dating messages ourselves across 12 categories. Claude drafted the first 120, which we used for development. A team member wrote the last 80 without ever seeing model output, and those were the final test."

## Slide 5: What the data told us

**On the slide:** three small charts: message length (dating median 9 words vs 34–36 in D1), identity words in D1 (31% of messages with "gay" labelled toxic vs 8% overall), and the D3 word weights (*you*, *my*, *me* push towards gaslighting).

**Script:** "Three things from the exploration shaped everything after. Dating messages are short, a median of nine words, so there's little evidence per message. In the toxic data, comments that mention identity groups are labelled toxic up to four times more often, which is a bias risk. And in the gaslighting data, everyday words like 'you' and 'my' count as gaslighting evidence, because the safe class is a therapist being reassuring. Remember that one; it comes back."

## Slide 6: Four models

**On the slide:** a four-row table: TF-IDF + logistic regression (baseline, 1 MB) → CNN over GloVe, filters over 3–5 words (ours, 20 MB) → DistilBERT frozen, only a new output layer → DistilBERT fine-tuned, all 66 M parameters (265 MB). The shared settings: weighted cross-entropy, Adam / AdamW, early stopping on validation, threshold chosen on validation.

**Script:** "The baseline counts words and phrases. Our CNN reads windows of three to five words using GloVe word vectors, so it sees word order. Then DistilBERT, a pretrained transformer, in two ways: frozen, where we only train a new output layer, and fine-tuned, where we train everything with a small learning rate. All four use class-weighted loss for the imbalance, early stopping, and a threshold picked on validation data only."

## Slide 7: Keeping the test honest

**On the slide:** a checklist. Duplicates and test-overlapping texts removed from D1 training (974 texts). D3 split by chat and its intent field dropped. 80 dating messages written blind. Every test set run once, on 18 November. Paired bootstrap intervals.

**Script:** "We were strict about leakage. We removed training texts that also appear in the test set, kept the gaslighting data split by conversation, and dropped a field that literally states the speaker's intent. Every test set ran exactly once, at the end, and we use paired bootstrap intervals to say whether one model is reliably better than another."

## Slide 8: Result 1, toxic language

**On the slide:** the toxic table (D1 F1, D2 F1, dating F1, dating ROC-AUC) with fine-tuned DistilBERT highlighted on D1 and frozen DistilBERT on dating. A strip: "Explicit insults 6/6 for every model. Put-downs 0–2/7. Threats 0–1/7."

**Script:** "On news comments the ranking is clear and reliable: fine-tuned DistilBERT 0.66, our CNN 0.62, the baseline 0.57, and the same holds on Wikipedia comments. On dating messages, every model catches every explicit insult but misses almost all put-downs and threats. The models learned rude words, not hostility: 'I despise you' scores 0.14, 'you are pathetic' scores 1. The only reliable winner on dating is frozen DistilBERT, plus 0.16 F1 over the baseline, maybe because it keeps more general language knowledge."

## Slide 9: Result 2, gaslighting

**On the slide:** two big numbers: "0.92–0.98 F1 on its own test" vs "ROC-AUC 0.33–0.58 on dating = chance". A side-by-side quote: a D3 therapist line vs a dating gaslighting line.

**Script:** "Gaslighting is our main finding. All eight gaslighting models score above 0.92 on their own test set, and on our dating messages all of them are at chance level. Why? The training data is a therapist undermining a patient, and the safe class is the same therapist being supportive. The models learned what therapy sounds like, not what gaslighting is. We tried adding everyday messages to fix false alarms. It cut them from 65% to 3%, but it also killed detection. No architecture fixes this; you need data from the real setting."

## Slide 10: Demo / prototype

**On the slide:** the app screenshot (`results/prototype_v1.jpg`) or a 30-second live demo.

**Script:** "Here's the prototype. You paste a chat and every message gets a toxic score and a grey experimental gaslighting score. The highlighted words drove the score. Look at this: 'work is a bit crazy' gets flagged because of 'crazy', and 'you are going to be sorry you ignored me' doesn't. That's exactly what our numbers predicted, so the app says clearly that it's a support tool and that no flag doesn't mean safe."

**Tip:** have the screenshot ready in case the live demo fails. The app starts in about 10 seconds with `python -m red_flag_detector.app`.

## Slide 11: Business case and responsible AI

**On the slide:** two columns.
- **Business:** user = dating-app T&S teams; Phase 1 paid pilot, triage of reported messages, a human decides; Phase 2 consented in-domain data for gaslighting and coercion detection; licence per platform or by volume. Market: dating apps earned more than $6B in 2024, Match Group alone $3.5B in 2025; the UK Online Safety Act covers dating services and lists coercive behaviour as illegal content. Existing tools (Tinder "Does This Bother You?", Hinge "Are You Sure?", Bumble's rude-message detection) catch rudeness, not gaslighting.
- **Responsible AI:** identity-word bias (1–2 of 7 identity messages flagged); privacy of pasted chats (nothing stored, GDPR and consent for Phase 2); risks of false reassurance, false alarms, and treating a score as proof.

**Script:** "So what can you sell? Not an automatic red-flag detector for daters; our numbers don't support that. But the toxic model ranks dating messages decently, which is useful for moderators: put likely abuse at the top of the queue, with a human always deciding. Phase two collects consented, reported messages, the in-domain data our results show is missing. On responsible AI: the training labels link identity words to toxicity, pasted chats contain someone else's messages, and a score is never proof of abuse."

## Slide 12: Conclusion and next steps

**On the slide:** three lines. "Explicit toxicity transfers. Subtle harm doesn't. Gaslighting doesn't at all." / "On dating messages, the data's setting matters more than the model." / Next steps: in-domain consented data, chat-level models, a fairness audit.

**Script:** "To answer our question: explicit toxic language transfers, subtle harm mostly doesn't, and gaslighting doesn't at all. Our CNN beats the baseline but not DistilBERT on the data it was trained on, and on dating messages the data's setting matters more than any model. Next we'd need real, consented dating data and models that read whole conversations. Thank you, and we're happy to take questions."

---

## Q&A preparation

Everyone should be able to answer all of these. As a default: Alp takes the model and training questions, Milan leakage and evaluation, Aysel data, Raya business, Sacha the app, and Alexander AI use and anything else.

| Likely question | Short answer |
|---|---|
| Why not accuracy? | D1 is 8% toxic; always saying "not toxic" scores 92% accuracy and F1 0. |
| How did you avoid data leakage? | Deduplicated D1 and removed 974 train texts found in the test file; D3 split by chat and dropped `internal thought`; thresholds and hyperparameters chosen on validation only; each test set run once; final dating messages written blind. |
| Why is frozen DistilBERT worst on D1 but best on dating? | Its [CLS] vector was never trained to summarise a message, so it underfits D1; but it isn't specialised to news comments, so it may transfer better. Based on 60 messages, so a hypothesis. |
| Why weighted cross-entropy? | The imbalance: without weights the model mostly predicts the majority class. |
| Why AdamW for fine-tuning but Adam elsewhere? | AdamW decouples weight decay from the gradient update, the standard recipe for transformers (with warm-up and linear decay). |
| Why does the CNN use unfold + linear instead of Conv1d? | Same output (a test checks it), but cuDNN's Conv1d backward pass was about 50× slower on our GPU. |
| How were the CNN hyperparameters chosen? | An 8-setting validation grid (frozen vs fine-tuned GloVe, two learning rates, with and without weight decay), selected on validation PR-AUC; filters 3/4/5 × 100 and dropout 0.5 follow Kim (2014). |
| Isn't a self-written test set biased? | Yes, which is a stated limit. We mitigated it: the final 80 were written without seeing model output and reviewed by the group, and we report per category with intervals. |
| Why not label twice and report kappa? | Time; we report it as a limitation and some labels are debatable. |
| What does ROC-AUC 0.5 mean? | The model ranks a harmful message above a harmless one no more often than a coin flip. |
| Why didn't the everyday-messages fix work? | It taught the model that ordinary chat is safe, but D3 still never showed it partner-style gaslighting, so it stopped flagging dating messages altogether. |
| How did you use AI? | Claude drafted code, ran training and drafted the docs and this report; we decided the scope, wrote the final test set and reviewed everything. See the declaration. |
| Is this ready for real users? | No, only as a moderator triage aid with a human deciding. |
