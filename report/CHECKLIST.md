# Submission checklist (due Tuesday 24 November, Blackboard)

The drafts in this folder were written by Claude on 18 November from the frozen results. They are a starting point: each owner rewrites their part, fills the **[TEAM: ...]** gaps and must be able to explain it in the Q&A.

| File | What it is |
|---|---|
| `REPORT_DRAFT.md` | Full report draft with all 10 required sections (about 5,200 words, roughly 11–13 pages with figures) |
| `SLIDES_AND_SCRIPT.md` | 12 slides, a timed 10-minute script and Q&A answers |
| `BUSINESS_SOURCES.md` | Sourced facts for the market, competitor and regulation parts. Open each link before quoting it |
| `../scripts/make_submission_zip.py` | Builds the code zip from committed files |

## Rubric check

| Criterion | Pts | Where it is covered | Still to do |
|---|---:|---|---|
| Problem, impact, business value | 1.5 | Report §2, §7; slides 2 and 11 | Check the sources, choose the market numbers, make the business model your own |
| Originality | 0.5 | Gaslighting classifier, our own dating test, the transfer question | Say it explicitly in the intro and on slide 3 |
| Data | 1.5 | Report §3; notebook 03; `results/eda_*.png` | Insert the figures in the PDF |
| Methodology and models | 2.0 | Report §4 | Nothing beyond review |
| Evaluation and results | 1.5 | Report §5 | **Error analysis: fill `team_cause`, counts and 3 examples per model (§5.4)** |
| Prototype and code | 1.5 | Report §6; README; app; 52 tests | Add screenshots to the PDF; decide on the README link (below) |
| Report | 0.5 | `REPORT_DRAFT.md` | **Contribution table (§9)**, AI declaration details (§10), export to PDF within 15 pages |
| Presentation | 1.0 | `SLIDES_AND_SCRIPT.md` | Build the deck, assign speakers, rehearse to 10:00 |

## Before submitting

1. Error analysis done and pasted into §5.4.
2. Every **[TEAM: ...]** bracket and the draft box at the top of the report removed.
3. Report exported to PDF, 10–15 pages, figures readable, contribution table filled honestly.
4. Slides exported to PDF or PPTX.
5. Code zip:
   - The README links to `PROJECT_GUIDE.md` ("New here? Read..."), which the zip leaves out. Delete that line, or keep the guide in the zip by removing it from `EXCLUDE` in the script.
   - Commit everything, then run `python scripts/make_submission_zip.py`. Check its output for warnings, then upload `dist/RedFlagDetector_Group7.zip`, or submit the GitHub link.
   - Tested on 18 November: the zip has 104 files and is 30.6 MB, all 52 tests pass from the extracted copy, and the app starts from it.
6. Nothing from `data/raw/`, no Kaggle files and no fine-tuned weights in any upload. The script only packs committed files, and Git ignores those.
7. Everyone has read the code they present (CLAUDE.md: models are frozen, do not retrain or rerun the final evaluation).
