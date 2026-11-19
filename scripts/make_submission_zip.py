"""Build the code zip for Blackboard from committed files only.

Usage (from the repo root, after committing):

    python scripts/make_submission_zip.py            # writes dist/RedFlagDetector_Group7.zip
    python scripts/make_submission_zip.py --list     # only print what would go in

Only files tracked by Git at HEAD are packed, so .venv, __pycache__, data/raw/,
fine-tuned DistilBERT weights and anything uncommitted can never end up in the zip.
Team-internal files listed in EXCLUDE stay out. Edit that list if the team decides
otherwise. The script warns when a kept Markdown file links to an excluded one.
"""

import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP_NAME = "RedFlagDetector_Group7.zip"
TOP_FOLDER = "RedFlagDetector_Group7"

# Paths (files or folder prefixes) that are for the team, not the professor.
EXCLUDE = [
    "CLAUDE.md",
    "PROJECT_GUIDE.md",
    "TEAM_UPDATE.md",
    "report/",  # report and slides are submitted separately as PDF / PPTX
    "scripts/",
    "results/final_dry_run/",
    ".gitignore",
]

# The Session 6 structure the rubric asks for.
REQUIRED = ["README.md", "pyproject.toml", "requirements.txt", "environment.yml", "src/red_flag_detector/", "notebooks/", "data/", "results/", "tests/"]


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    return [p for p in out.stdout.decode("utf-8").split("\0") if p]


def is_excluded(path: str) -> bool:
    return any(path == rule or (rule.endswith("/") and path.startswith(rule)) for rule in EXCLUDE)


def broken_links(kept: list[str]) -> list[str]:
    """Relative Markdown links in kept files that point at excluded files."""
    kept_set = set(kept)
    problems = []
    for path in kept:
        if not path.endswith(".md"):
            continue
        text = (ROOT / path).read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)#\s]+)", text):
            if "://" in target:
                continue
            resolved = (Path(path).parent / target).as_posix()
            resolved = str(Path(resolved)).replace("\\", "/")
            if resolved not in kept_set and not any(k.startswith(resolved.rstrip("/") + "/") for k in kept_set):
                problems.append(f"{path} links to {target}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="print the files without writing the zip")
    args = parser.parse_args()

    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    if status.strip():
        print("Warning: you have uncommitted changes; only committed files go into the zip.\n")

    files = tracked_files()
    kept = [p for p in files if not is_excluded(p)]
    missing = [r for r in REQUIRED if not any(p == r or p.startswith(r) for p in kept)]
    links = broken_links(kept)

    if args.list:
        print("\n".join(kept))
    print(f"\n{len(kept)} files kept, {len(files) - len(kept)} left out.")
    if missing:
        print("Missing from the Session 6 structure: " + ", ".join(missing))
    for problem in links:
        print("Broken link after cleanup: " + problem)
    if args.list:
        return 1 if missing else 0

    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    target = dist / ZIP_NAME
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in kept:
            archive.write(ROOT / path, f"{TOP_FOLDER}/{path}")
    size_mb = target.stat().st_size / 1e6
    print(f"Wrote {target.relative_to(ROOT)} ({size_mb:.1f} MB).")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
