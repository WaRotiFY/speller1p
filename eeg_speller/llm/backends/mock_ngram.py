"""Deterministic interpolated character n-gram baseline."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import math
import re
import numpy as np
from eeg_speller.core.distributions import SymbolDistribution, log_probs
from eeg_speller.text.normalize import normalize


def _context(text: str) -> str:
    """normalize() strips edge spaces, but a trailing space is a word boundary here."""
    clean = normalize(text)
    if clean and text[-1:].isspace() and not clean.endswith(" "):
        clean += " "
    return clean


class MockNGram:
    """Source Model (Fast LLM) mock; NEED-F-06/08; MA-08/10/11; Q-13."""
    version = "1"

    def __init__(self, cfg: dict, corpus_path: str | Path):
        # IMPROV[Q-34]
        self.cfg = cfg
        raw = Path(corpus_path).read_bytes()
        self.weights_hash = hashlib.sha256(raw).hexdigest()
        text = normalize(raw.decode("utf-8"))
        self.order = int(cfg["mock"]["order"])
        self.discount = float(cfg["mock"]["discount"])
        self.model_id = cfg["model_id"]
        self.counts = [defaultdict(Counter) for _ in range(self.order)]
        continuation_sets = [defaultdict(lambda: defaultdict(set)) for _ in range(self.order - 1)]
        self.vocab = Counter(re.findall(r"[а-я]+", text))
        for i, ch in enumerate(text):
            for n in range(self.order):
                context = text[max(0, i - n):i]
                self.counts[n][context][ch] += 1
                if n < self.order - 1 and i - n - 1 >= 0:
                    continuation_sets[n][context][ch].add(text[i - n - 1])
        self.continuations = [{context: Counter({ch: len(left) for ch, left in endings.items()})
                               for context, endings in table.items()} for table in continuation_sets]

    def _dist(self, context: str, symbols: tuple[str, ...]):
        # ASSUMPTION[MA-10]
        base = self.continuations[0].get("", self.counts[0][""])
        total = sum(base.values())
        p = np.array([(base.get(s, 0) + 1) / (total + len(symbols)) for s in symbols])
        for n in range(1, self.order):
            table = self.counts[n] if n == self.order - 1 else self.continuations[n]
            counts = table.get(context[-n:])
            if not counts:
                continue
            denom = sum(counts.values())
            d = self.discount
            p = np.array([max(counts.get(s, 0) - d, 0) / denom for s in symbols]) + d * len(counts) / denom * p
        return p / p.sum()

    def predict(self, context, repertoire):
        # ASSUMPTION[MA-08]
        probs = self._dist(_context(context), repertoire.symbols)
        bs = float(self.cfg["char_marginal"]["backspace_prob"])
        probs *= 1 - bs
        probs[repertoire.index("⌫")] = bs
        return SymbolDistribution(repertoire, log_probs(probs), f"fast_llm:{self.model_id}", len(context))

    def top_words(self, context: str, n: int):
        # ASSUMPTION[MA-11]
        clean = _context(context)
        prefix = clean.split(" ")[-1] if clean and clean[-1] not in " .," else ""
        candidates = [w for w in self.vocab if w.startswith(prefix) and len(w) > len(prefix)]
        scored = []
        for word in candidates:
            state = clean
            score = 0.0
            symbols = tuple("абвгдежзийклмнопрстуфхцчшщъыьэюя .,")
            for ch in word[len(prefix):] + " ":
                p = self._dist(state, symbols)
                # Use a floor if the candidate character is outside the model vocabulary.
                score += math.log(max(float(p[symbols.index(ch)]), 1e-12))
                state += ch
            scored.append((word, score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        scored = scored[:n]
        if not scored:
            return []
        maximum = max(score for _, score in scored)
        masses = [math.exp(score - maximum) for _, score in scored]
        z = sum(masses)
        return [(word, mass / z) for (word, _), mass in zip(scored, masses)]
