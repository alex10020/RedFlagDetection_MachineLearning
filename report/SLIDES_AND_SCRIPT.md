# Slides and 10-minute script

> **DRAFT (20 November).** 12 slides, about 1,100 spoken words, roughly 10 minutes at a calm pace. The talk is built around a game: on slide 1 the class votes on two dating messages, and the answers come back on slides 8 and 9. Six speakers with two slides each, matching the contribution table in the report. Each speaker should reword their lines so they sound like themselves, then time a full run. If it runs long, shorten slide 7 first.
>
> Visual style: the colours of the 5 November check-in deck, with chat bubbles and red flags as the recurring motif.

| # | Slide | Speaker | Time |
|---|---|---|---:|
| 1 | Red flag or not? (the game) | Alexander | 0:45 |
| 2 | Most harm doesn't use rude words | Raya | 0:50 |
| 3 | Our question | Alexander | 0:35 |
| 4 | Where our models learned | Aysel | 0:50 |
| 5 | Three clues in the data | Aysel | 0:45 |
| 6 | Meet the four contestants | Alp | 0:55 |
| 7 | No cheating: the test stayed sealed | Milan | 0:35 |
| 8 | Result 1: it catches insults and misses hostility | Alp | 1:05 |
| 9 | Result 2: gaslighting | Milan | 1:05 |
| 10 | Try it: the app | Sacha | 1:05 |
| 11 | Business case and responsible AI | Raya | 0:55 |
| 12 | What we learned | Sacha | 0:35 |
| | **Total** | | **10:00** |

---

## Slide 1: Red flag or not?

**On the slide:** "Red flag or not?" Two chat bubbles:
- A: "you are going to be sorry you ignored me" (a threat)
- B: "so when do i get to beat you at mini golf?" (harmless)

**Script:** "Hi, we're Group 7, and we'd like to start with a quick game. Here are two messages from a dating chat. Hands up if you'd flag message A as a red flag. And message B? Keep your answers in mind. We gave both messages to the AI models we built, and later in the talk you'll see how they voted."

**Tip:** the two messages were picked because the models get them wrong. If anyone asks, say so: they show the typical mistakes, and the models do catch every explicit insult (slide 8). Scores: A 0.21 from the toxic CNN in the app; B 1.00 from the CNN trained on D3 only (fine-tuned DistilBERT: 0.999), on the development set.

## Slide 2: Most harm doesn't use rude words

**On the slide:** a scale from loud to quiet: explicit insult, put-down, threat, guilt-tripping, sexual pressure, gaslighting. Two numbers: 48% of US online daters have been harassed (Pew Research Center, 2023); 53% of women on dating apps were gaslit, a harm reported far less (BIT / Minderoo, 2026). Our user: the moderator who reads every report.

**Script:** "Harassment on dating apps is common: 48% of US online daters have experienced it. A lot of it is quiet, though. A put-down dressed up as a joke, a threat in polite words, guilt-tripping, or gaslighting, which means making someone doubt their own memory. In a 2026 survey of women on dating apps, 53% had been gaslit, and that kind of harm gets reported far less than insults. Every report that does come in has to be read by a trust and safety moderator, and those moderators are who we built this for."

## Slide 3: Our question

**On the slide:** "Can models trained outside dating spot toxic language and gaslighting in dating messages? And does our own neural network keep up with a big pretrained one?" Three steps: learn from other text, test on dating messages, build an app.

**Script:** "No public dataset of real dating chats exists, for good privacy reasons. So we asked: can models trained on text from outside dating still spot toxic language and gaslighting in dating messages? And can our own neural network keep up with a big pretrained one? We trained on other data, tested on dating messages, and built an app you can try."

## Slide 4: Where our models learned

**On the slide:** three cards. 1.75 M news-site comments, 8% toxic, for toxic language. 2,000 ChatGPT-written therapy chats, for gaslighting (Li et al., 2025). 200 dating messages we wrote ourselves, 80 locked away for one final test. Strip: always answering "fine" would be 92% accurate and catch nothing, so we measure F1.

**Script:** "For toxic language we used 1.75 million comments from news websites. Only 8% of them are toxic, so a model that always says 'fine' would be 92% accurate and catch nothing. We measure F1 instead. For gaslighting, the only labelled data we found is 2,000 conversations that ChatGPT wrote for a research paper. To test on dating, we wrote 200 dating messages ourselves, in 12 categories from flirting to threats. A teammate wrote the last 80 without seeing any model output, and we locked them away for one final test."

## Slide 5: Three clues in the data

**On the slide:** "9 words": a typical dating message, against about 35 in the news comments. "4×": messages mentioning "gay" were labelled toxic 31% of the time, against 8% overall. "you, my, me": count as evidence of gaslighting in the ChatGPT data.

**Script:** "The data gave us three warnings. Dating messages are short: a typical one is nine words, against about 35 in the news comments, so a model has little to go on. In the news comments, messages that mention 'gay' are labelled toxic 31% of the time, against 8% overall, so a model can learn the word itself as an insult. That's a bias risk. And in the gaslighting data, ordinary words like 'you', 'my' and 'me' count as evidence of gaslighting. Remember that one; it comes back."

## Slide 6: Meet the four contestants

**On the slide:** four cards. Word Counter (TF-IDF + logistic regression, our baseline). Phrase Reader (our CNN over GloVe word vectors, built by us). Bookworm (DistilBERT, frozen). Specialist (DistilBERT, fine-tuned). Each with what it reads and its training time. Strip: same data, same rules, trained for both topics.

**Script:** "We trained four contestants on each topic. The Word Counter is our baseline: it counts words and short phrases. The Phrase Reader is the neural network we built, a CNN that reads three to five words at a time, so it sees some word order. The other two use DistilBERT, a pretrained language model that has already read books and Wikipedia. The Bookworm keeps that knowledge frozen, and we only train its last layer. The Specialist gets retrained completely on our data. All four got the same data and the same rules, and we chose every setting on validation data."

## Slide 7: No cheating: the test stayed sealed

**On the slide:** a sealed envelope: 80 dating messages, written blind, opened once on 18 November. Three rules: 974 training texts removed because they were also in the test set; settings picked on validation data; one final run, nothing tuned afterwards.

**Script:** "It's easy to fool yourself in machine learning, so we set three rules. We removed 974 training texts that also appeared in the test set. We picked every threshold and setting on validation data. And our 80 final dating messages stayed sealed until a single run on 18 November. Everything on the next slides comes from that one run; we didn't tune anything afterwards."

## Slide 8: Result 1: it catches insults and misses hostility

**On the slide:** a bar chart of F1 on news comments (Specialist 0.66, Phrase Reader 0.62, Word Counter 0.57, Bookworm 0.43). Caught on the final dating messages: explicit insults 6 of 6 for every model, put-downs at most 2 of 7, threats at most 1 of 7. The answer to A. "I despise you" 0.14 against "you are pathetic" 1.00.

**Script:** "First, toxic language. On news comments the Specialist wins with 0.66, and our Phrase Reader comes second, reliably ahead of the baseline. On dating messages, every model catches every explicit insult, but almost all put-downs and threats get through. Back to the game: message A is a threat, and it scored 0.21, so no flag. 'I despise you' scores 0.14, while 'you are pathetic' scores 1. The models react to rude words, and hostility in polite words slips past them. On dating messages, only the Bookworm did reliably better than the baseline."

## Slide 9: Result 2: gaslighting

**On the slide:** "0.92–0.98": F1 on the ChatGPT chats, all eight gaslighting models. "A coin flip": on dating messages (ROC-AUC 0.33–0.58). The answer to B, and a real gaslighting message that scored 0.005. The cause: the models learned what a ChatGPT therapist sounds like.

**Script:** "Gaslighting gave us our biggest finding. On their own test set, all eight gaslighting models scored between 0.92 and 0.98. On our dating messages they did no better than a coin flip. Back to the game: message B, the mini golf one, is harmless, and the model gave it the maximum gaslighting score. Real gaslighting, like 'you're making a big deal out of nothing, as usual', scored 0.005. The training data is ChatGPT playing a therapist who undermines a patient, so the models learned what therapy sounds like: questions, and words like 'you', 'my' and 'me'. One model flagged 59 of 60 harmless dating messages. We tried adding everyday messages: false alarms dropped from 65% to 3%, and the model stopped catching gaslighting altogether. Fixing this would take gaslighting examples from real dating chats."

## Slide 10: Try it: the app

**On the slide:** the app screenshot and three steps: paste a chat; each message gets a toxic score that can raise a red flag, with the words behind it highlighted; the gaslighting score is grey and never flags.

**Script:** "Here's our prototype. You paste a chat, and every message gets a toxic score. Above the threshold it raises a red flag and highlights the words that pushed the score up, so you can see why. The gaslighting score is shown in grey and never raises a flag, because, as you just saw, it doesn't work on dating messages. Let me show you. [live demo] 'Stupid tease' and 'you're pathetic' are flagged, which is right. 'Work is a bit crazy' is flagged too, because of the word 'crazy', and the threat from our game isn't flagged. So the app tells users that no flag doesn't mean safe."

**Tip:** start the app before the talk with `python -m red_flag_detector.app` (about 10 seconds) and paste the example chat from the screenshot. If the demo fails, talk over the screenshot.

## Slide 11: Business case and responsible AI

**On the slide:** Business: sell to trust and safety teams, not to daters; Phase 1 ranks reported messages for moderators; Phase 2 learns gaslighting from consented, reported messages; dating apps earned more than $6 B in 2024; Tinder, Hinge and Bumble catch rudeness, and none claims to catch gaslighting. Responsible AI: a person decides every case; identity-word bias; privacy (nothing stored, consent under GDPR); a score is never proof.

**Script:** "Our results don't support an app that tells daters their match is toxic. What works today is ranking reported messages so moderators see likely abuse first, with a person making every decision. Dating apps earned over 6 billion dollars in 2024. Tinder, Hinge and Bumble already catch rude messages, and none of them claims to catch gaslighting. That's phase two: learning from messages users report, with their consent. On responsible AI: the training labels link identity words to toxicity, and a pasted chat contains someone else's words, so the app stores nothing. And a score is never proof of abuse."

## Slide 12: What we learned

**On the slide:** "AI catches insults, misses manipulation." Explicit insults transfer; subtle harm is mostly missed; gaslighting doesn't transfer; on dating messages the data matters more than the model. The game score: models 0 of 2. Next steps: real, consented dating data; models that read the whole chat. "Thank you. Questions?"

**Script:** "To answer our question: explicit insults transfer to dating, most subtle harm doesn't, and gaslighting doesn't transfer at all. Our CNN beat the baseline, but on dating messages the training data mattered far more than the choice of model. In our game the models scored zero out of two. We picked hard messages on purpose, because those are the ones a moderator needs to see. Next we'd need real, consented dating data and models that read the whole conversation. Thank you, we're happy to take questions."

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
| Why those two game messages? | We picked them because the models get them wrong, to show the two typical mistakes: a threat in polite words that the toxic model misses, and ordinary chat that the gaslighting model flags. The models do catch every explicit insult (6 of 6). |
| How did you use AI? | Claude drafted the code, ran the training and drafted the first 120 dating messages; we decided the scope, wrote the final test set and reviewed everything. See the declaration. |
| Is this ready for real users? | No, only as a moderator triage aid with a human deciding. |
