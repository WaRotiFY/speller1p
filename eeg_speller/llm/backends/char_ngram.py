"""Pretrained character n-gram (lookup table) — the default fast model.

A model file is produced by ``scripts/train_char_ngram.py``: P(next char | previous
order-1 chars) over the LM alphabet plus a word list for next-word candidates.
Prediction is one table lookup, so it fits any epoch length.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from eeg_speller.core.distributions import SymbolDistribution, log_probs
from eeg_speller.core.keyboard import LM_ALPHABET, keyboard_text, project

PROJECT = Path(__file__).resolve().parents[3]
MODELS = PROJECT / "models" / "ngram"
DEFAULT_MODEL = MODELS / "ru_char4.npz"


def available_models() -> list[dict]:
    """Model files in models/ngram with their metadata (for the operator window)."""
    out = []
    for path in sorted(MODELS.glob("*.npz")):
        try:
            with np.load(path) as data:
                meta = json.loads(str(data["meta"]))
        except Exception:
            continue
        out.append({"file": path.name, "order": meta.get("order"),
                    "train_chars": meta.get("train_chars"), "bpc": meta.get("bits_per_char_tasks"),
                    "sources": [s.get("kind") for s in meta.get("sources", [])]})
    return out


class CharNGram:
    """Source Model (Fast LLM): pretrained character n-gram; NEED-F-06/08; MA-08/10/11; Q-13."""
    version = "char-ngram@1"

    def __init__(self, cfg: dict):
        # IMPROV[Q-39]
        name = cfg.get("model_file") or DEFAULT_MODEL.name
        path = Path(name)
        if not path.is_absolute():
            path = MODELS / path
        if path.parent.resolve() != MODELS.resolve() and not Path(name).is_absolute():
            raise ValueError("n-gram model must be a file in models/ngram")
        if not path.is_file():
            raise ValueError(f"n-gram model not found: {path.name} (run scripts/train_char_ngram.py)")
        raw = path.read_bytes()
        self.weights_hash = hashlib.sha256(raw).hexdigest()
        with np.load(path) as data:
            self.table = data["table"].astype(np.float32)
            self.vocab = [str(w) for w in data["vocab"]]
            counts = data["vocab_count"].astype(np.float64)
            self.meta = json.loads(str(data["meta"]))
        if self.meta.get("alphabet") != LM_ALPHABET:
            raise ValueError("n-gram model alphabet does not match the speller LM alphabet")
        self.order = int(self.meta["order"])
        self.cfg = cfg
        self.model_id = cfg.get("model_id") or path.stem
        self.file = path.name
        self.V = len(LM_ALPHABET)
        self.index = {c: i for i, c in enumerate(LM_ALPHABET)}
        self.log_unigram = np.log(counts / counts.sum())
        self.by_first: dict[str, list[int]] = {}
        for i, w in enumerate(self.vocab):
            self.by_first.setdefault(w[:1], []).append(i)

    # ------------------------------------------------------------------ helpers
    def _clean(self, context: str) -> str:
        text = keyboard_text(context)
        text = re.sub(r" +", " ", text)
        return " " * (self.order - 1) + text

    def _row(self, history: str) -> np.ndarray:
        ctx = 0
        for ch in history[-(self.order - 1):]:
            ctx = ctx * self.V + self.index[ch]
        return self.table[ctx]

    def lm_distribution(self, context: str) -> np.ndarray:
        p = self._row(self._clean(context)).astype(np.float64) + 1e-7
        return p / p.sum()

    # --------------------------------------------------------------- interface
    def predict(self, context, repertoire):
        # ASSUMPTION[MA-08]
        cm = self.cfg.get("char_marginal", {})
        p = project(self.lm_distribution(context), repertoire.symbols,
                    float(cm.get("backspace_prob", 0.02)), float(self.cfg.get("other_prob", 0.005)))
        return SymbolDistribution(repertoire, log_probs(p), f"fast_llm:{self.model_id}", len(context))

    def top_words(self, context: str, n: int):
        # ASSUMPTION[MA-11]
        clean = self._clean(context)
        prefix = re.search(r"[а-я]*$", clean).group()
        history = clean[:len(clean) - len(prefix)]
        pool = self.by_first.get(prefix[:1], []) if prefix else range(len(self.vocab))
        cands = [i for i in pool if self.vocab[i].startswith(prefix) and len(self.vocab[i]) > len(prefix)]
        # Pre-rank by frequency, then score the best ones with the character model.
        cands.sort(key=lambda i: -self.log_unigram[i])
        scored = []
        cache: dict[str, np.ndarray] = {}
        for i in cands[:400]:
            word = self.vocab[i]
            state = history + prefix
            score = 0.0
            for ch in word[len(prefix):] + " ":
                key = state[-(self.order - 1):]
                row = cache.get(key)
                if row is None:
                    row = cache[key] = self._row(state)
                score += math.log(max(float(row[self.index[ch]]), 1e-9))
                state += ch
            scored.append((word, score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        scored = scored[:n]
        if not scored:
            return []
        top = max(s for _, s in scored)
        mass = [math.exp(s - top) for _, s in scored]
        z = sum(mass)
        return [(w, m / z) for (w, _), m in zip(scored, mass)]
