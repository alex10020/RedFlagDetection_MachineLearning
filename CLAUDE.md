Read @PROJECT_GUIDE.md before working on this project.

- Use the project's `.venv` (Python 3.13). Training runs on the local RTX 2070 GPU.
- The final run happened on 18 November (commit `4add43c`, results in `results/final/`). The models are frozen: never retrain them, change their thresholds or rerun `python -m red_flag_detector.final`.
- Never score dating messages S121 to S200 or the official D1, D2 and D3 test sets again outside `results/final/`.
- Never commit anything in `data/raw/`, the Kaggle files or fine-tuned DistilBERT weights.
- Branch from `main`, and push only after the user agrees.
- Record every change from the proposal and every new limitation in PROJECT_GUIDE.md under "Changes from the proposal" or "Limits to discuss in the report".
