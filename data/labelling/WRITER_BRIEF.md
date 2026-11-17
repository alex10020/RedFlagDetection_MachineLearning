# Brief: writing the 80 new dating messages

These 80 messages become our final dating test. The first 120 are a development set because we have seen the models' mistakes on them. So the new ones only count if nobody tunes anything on them.

## Rules

1. Write them yourselves, in your own words. Do not use ChatGPT, Claude or any other AI tool, and do not paraphrase the first 120 messages. Read the category descriptions below instead of the old messages.
2. Never run a model on them, paste them into the app or look at scores, until the final test run on Sunday 22 November.
3. Write messages that sound like a real dating-app chat. Vary length: some 3 words, some 25. Typos, slang, emoji and lowercase are fine.
4. No real chats, real names of people you know, phone numbers or other personal data. Invented first names are fine.
5. Harmful messages can be offensive where the category needs it. Do not use slurs: we do not need them to test the models, and they would end up in our repo.
6. Label each message as you write it. Two other people will label all 200 messages blind afterwards, so your label is a draft.

## How many per category

Half harmful, half not, like the first 120. Each writer writes about half of every category, so each category mixes two writing styles. The harder categories get the extra message.

| Category | Topic | Label | What it means | Count |
|---|---|---:|---|---:|
| `explicit_insult` | toxic | 1 | Direct insult with rude words | 6 |
| `put_down` | toxic | 1 | Belittling without rude words ("cute that you think you're smart") | 7 |
| `threat` | toxic | 1 | Harm stated or implied, including "you'll regret this" | 7 |
| `gaslighting` | gaslighting | 1 | Denying what happened, questioning the other person's memory or sanity, reframing their feelings as overreaction | 7 |
| `guilt_manipulation` | other_harm | 1 | Guilt-tripping or control: demanding replies, policing who they see | 7 |
| `sexual_pressure` | other_harm | 1 | Pushing for sexual content or meeting after a no or a hesitation | 6 |
| `friendly_flirt` | harmless | 0 | Warm interest, compliments | 6 |
| `neutral_logistics` | harmless | 0 | Plans, times, places | 6 |
| `playful_banter` | harmless | 0 | Teasing with insult-like words between people who clearly like each other | 7 |
| `identity_mention` | harmless | 0 | Mentions religion, ethnicity, sexuality or gender in a neutral or positive way | 7 |
| `healthy_conflict` | harmless | 0 | Disagreeing or setting a boundary respectfully | 7 |
| `polite_rejection` | harmless | 0 | Saying no kindly | 6 |
| **Total** | | | | **80** |

## Format

Add the rows to `data/dating_stress_test.csv` in one pull request, with ids `S121` to `S200`:

```
id,text,label,category,topic,label_status
S121,"<your message, in quotes if it has a comma>",0,polite_rejection,harmless,draft_team
```

Use `draft_team` as `label_status`, so we can tell your messages from the AI-drafted first 120. Put quotes around any text that contains a comma.

## After you finish

The two labellers, who are not the writers, run:

```bash
python -m red_flag_detector.labelling sheets
```

and follow the steps at the top of `src/red_flag_detector/labelling.py`.
