"""Dictionary words the speller could have meant, from the letter alternatives.

Every epoch of a P300 speller types exactly one symbol, so a misrecognized word has the
same length as the intended one: the typing errors are substitutions. For a typed word
and the posterior alternatives of each of its letters, every dictionary word of the same
length gets

    score(w) = sum_i log P(letter i = w_i) + beta * log freq(w)

where letters outside the recorded top-K share a small floor probability. The best few
words with their normalized share are shown to the slow LLM, which then only has to pick
by meaning instead of spelling a correction letter by letter (MA-37).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VOCAB = ROOT / "models" / "ngram" / "ru_char4.npz"
FLOOR = 0.003            # probability of a letter that was not among the top-K alternatives
BETA = 0.5               # weight of the word frequency (the letter posteriors already contain an LM prior)


@lru_cache(maxsize=2)
def _vocab(path: str):
    data = np.load(path, allow_pickle=False)
    words = [str(w) for w in data["vocab"]]
    logf = np.log(np.maximum(data["vocab_count"].astype(np.float64), 1.0))
    by_len: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for n in {len(w) for w in words}:
        idx = [i for i, w in enumerate(words) if len(w) == n]
        chars = np.array([[ord(c) for c in words[i]] for i in idx], dtype=np.int32)
        by_len[n] = (chars, np.array(idx))
    return words, logf, by_len


def candidates(word: str, letters, k: int = 4, vocab: Path = DEFAULT_VOCAB, beta: float = BETA,
               floor: float = FLOOR) -> list[tuple[str, float]]:
    """Up to ``k`` (word, share) pairs; the typed word itself is included when it is a word."""
    # ASSUMPTION[MA-37]
    if not word or not letters or len(letters) != len(word):
        return []
    words, logf, by_len = _vocab(str(vocab))
    if len(word) not in by_len:
        return []
    chars, idx = by_len[len(word)]
    score = beta * logf[idx]
    for j, alts in enumerate(letters):
        table = {ord(s): float(p) for s, p in (alts or []) if len(s) == 1}
        table.setdefault(ord(word[j]), max(table.values(), default=0.5))
        col = chars[:, j]
        lp = np.full(len(col), np.log(floor))
        for code, p in table.items():
            lp[col == code] = np.log(max(p, floor))
        score = score + lp
    top = np.argsort(-score)[:k]
    s = score[top]
    share = np.exp(s - s.max())
    share /= share.sum()
    return [(words[idx[i]], float(q)) for i, q in zip(top, share)]


def format_candidates(window: list[str], letters, k: int = 4) -> str:
    """Prompt lines 'typed: cand1 61, cand2 30, …' for the words of a correction window."""
    if not letters:
        return ""
    lines = ["Похожие слова из словаря:"]
    for word, chars in zip(window, letters):
        cands = candidates(word, chars, k)
        if cands:
            lines.append(f"{word}: " + ", ".join(f"{w} {round(q * 100)}" for w, q in cands))
    return "\n".join(lines) if len(lines) > 1 else ""


class DictionaryCorrection:
    """Slow-LLM stand-in without a language model: the best dictionary word when it is clear.

    A word is replaced by its first candidate only when that candidate holds at least
    ``min_share`` of the candidates' mass; otherwise it is kept as typed. Instant, local,
    and it never changes the number or the length of words.
    """
    version = "dictionary-letters@1"
    uses_letters = True

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_id = cfg.get("model_id") or "dictionary-letters-1"
        self.min_share = float(cfg.get("min_share", 0.8))
        self.prompt_hash = None
        self.last_messages = None
        self.last_reply = None

    def correct(self, left_context: str, window: list[str], letters=None) -> list[str]:
        out = []
        for i, word in enumerate(window):
            cands = candidates(word, letters[i]) if letters and i < len(letters) else []
            out.append(cands[0][0] if cands and cands[0][1] >= self.min_share else word)
        self.last_reply = " ".join(out)
        return out
