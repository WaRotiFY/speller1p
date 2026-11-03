"""Dictionary noisy-channel correction baseline."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import re
from eeg_speller.text.normalize import words


def edit_cost(a: str, b: str, same_row_col_cost: float = 0.5) -> float:
    from eeg_speller.core.language import lm_alphabet
    layout = lm_alphabet() + "⌫"
    pos = {ch: divmod(i, 6) for i, ch in enumerate(layout)}
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            if ca == cb:
                sub = 0
            elif ca in pos and cb in pos and (pos[ca][0] == pos[cb][0] or pos[ca][1] == pos[cb][1]):
                sub = same_row_col_cost
            else:
                sub = 1
            current.append(min(prev[j] + 1, current[j-1] + 1, prev[j-1] + sub))
        prev = current
    return prev[-1]


class MockCorrection:
    """Post-factum Text Correction mock; NEED-F-07; MA-17/18; Q-14."""
    version = "1"

    def __init__(self, cfg: dict, corpus_path: str | Path):
        # ASSUMPTION[MA-17]
        # STUB[Q-14]
        self.cfg = cfg
        raw = Path(corpus_path).read_bytes()
        self.weights_hash = hashlib.sha256(raw).hexdigest()
        self.model_id = cfg["model_id"]
        tokens = words(raw.decode("utf-8"))
        self.freq = Counter(tokens)
        self.bigram = defaultdict(Counter)
        for left, right in zip(tokens, tokens[1:]):
            self.bigram[left][right] += 1

    def correct(self, left_context: str, window: list[str]) -> list[str]:
        # ASSUMPTION[MA-18]
        out = []
        previous = words(left_context)[-1] if words(left_context) else ""
        max_edit = self.cfg["mock"]["max_edit"]
        same = self.cfg["mock"]["same_row_col_cost"]
        for observed in window:
            if observed in self.freq:
                out.append(observed)
                previous = observed
                continue
            candidates = [observed]
            candidates += [w for w in self.freq if abs(len(w)-len(observed)) <= max_edit
                           and edit_cost(observed, w, same) <= max_edit]
            def score(w):
                import math
                prior = self.bigram[previous].get(w, 0) + 1
                unigram = self.freq.get(w, 0) + 1
                return math.log(prior) + 0.8 * math.log(unigram) - 2.5 * edit_cost(observed, w, same)
            selected = max(set(candidates), key=lambda w: (score(w), w))
            out.append(selected)
            previous = selected
        return out
