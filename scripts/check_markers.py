"""Fail when a documented MA/Q has no code marker or a marker is undocumented."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def check():
    assumptions = (ROOT / "docs" / "research" / "ASSUMPTIONS.md").read_text(encoding="utf-8")
    questions = (ROOT / "docs" / "research" / "OPEN_QUESTIONS.md").read_text(encoding="utf-8")
    code = "\n".join(p.read_text(encoding="utf-8") for folder in ("eeg_speller", "stimulus_kit", "inference")
                     for p in (ROOT / folder).rglob("*.py"))
    known_ma = set(re.findall(r"^### (MA-\d\d)\.", assumptions, re.M))
    known_q = set(re.findall(r"^### (Q-\d\d)\.", questions, re.M))
    marked_ma = set(re.findall(r"# ASSUMPTION\[(MA-\d\d)\]", code))
    marked_q = set(re.findall(r"# (?:STUB|IMPROV)\[(Q-\d\d)\]", code))
    errors = []
    for kind, known, marked in (("MA", known_ma, marked_ma), ("Q", known_q, marked_q)):
        errors += [f"{kind} undocumented marker: {item}" for item in sorted(marked - known)]
        errors += [f"{kind} missing code marker: {item}" for item in sorted(known - marked)]
    return errors


if __name__ == "__main__":
    errors = check()
    print("\n".join(errors) if errors else "All MA/Q markers match documentation.")
    sys.exit(bool(errors))
