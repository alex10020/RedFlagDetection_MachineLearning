# Submission checklist (due Tuesday 24 November, Blackboard)

Claude wrote the drafts in this folder from the frozen results (18 November). Each owner below rewrites their part in their own words, fills the remaining **[TEAM: ...]** gaps, and must be able to explain it in the Q&A.

| File | What it is |
|---|---|
| `REPORT_DRAFT.md` | Full report draft with all 10 required sections, error analysis included |
| `SLIDES_AND_SCRIPT.md` | 12 slides, two per speaker, with a timed 10-minute script and Q&A answers |
| `BUSINESS_SOURCES.md` | Market, harassment, competitor and regulation facts, each checked against its source page |
| `../results/errors/team_cause_counts.csv` | Error-analysis counts per model and cause |
| `../scripts/make_submission_zip.py` | Builds the code zip from committed files |

## Who does what

| Person | Report | Slides | Due |
|---|---|---|---|
| Alexander | §1 executive summary, §8 conclusions, final edit; code zip and Blackboard upload | 1, 3 | Tue 24 Nov |
| Alp | §4 methodology, §5 results (check against the code) | 6, 8 | Sat 21 Nov |
| Raya | §2 business case (sources already checked; make the business model your own) | 2, 11 | Sat 21 Nov |
| Milan | §7 responsible AI; second reader: check 20 rows of `team_cause` and note any changes | 7, 9 | Sat 21 Nov |
| Aysel | §3 data and EDA figures; slide deck design in the check-in style | 4, 5 | Sun 22 Nov |
| Sacha | §6 prototype with your own screenshots, the live demo; PDF assembly (figures, references, 15-page limit) | 10, 12 | Mon 23 Nov |

## Rubric check

| Criterion | Pts | Covered in | Status |
|---|---:|---|---|
| Problem, impact, business value | 1.5 | Report §2, §7; slides 2, 11 | Drafted with checked sources |
| Originality | 0.5 | Gaslighting classifier, own dating test, transfer question | State it in §1 and on slide 3 |
| Data | 1.5 | Report §3; notebook 03; `results/eda_*.png` | Drafted; insert figures in the PDF |
| Methodology and models | 2.0 | Report §4 | Drafted |
| Evaluation and results | 1.5 | Report §5, including §5.4 error analysis | Drafted; Milan's second read pending |
| Prototype and code | 1.5 | Report §6; README; app; 52 tests | Done; add screenshots to the PDF |
| Report | 0.5 | `REPORT_DRAFT.md` | Contribution table filled with assigned tasks; export to PDF |
| Presentation | 1.0 | `SLIDES_AND_SCRIPT.md` | Build the deck, rehearse to 10:00 |

## Before submitting

1. Every **[TEAM: ...]** bracket and the draft box at the top of the report removed.
2. The contribution table (§9) describes what each person actually did.
3. Report exported to PDF. The draft is about 6,600 words plus tables. If the PDF runs over 15 pages, shorten §5.4 to the two tables, the summary and the examples table first, then §3.3.
4. Slides exported to PDF or PPTX, and rehearsed to 10 minutes.
5. Code zip: commit everything, run `python scripts/make_submission_zip.py`, check that it prints no warnings, and upload `dist/RedFlagDetector_Group7.zip` (or submit the GitHub link). Tested on 18 November: all 52 tests pass and the app starts from the extracted zip.
6. Nothing from `data/raw/`, no Kaggle files and no fine-tuned weights in any upload. The script only packs committed files, and Git ignores those.
7. Everyone has read the code they present. The models are frozen: do not retrain them or rerun the final evaluation.
