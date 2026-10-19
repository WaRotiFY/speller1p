"""Compare fast language models on data/tasks_ru.txt (a small start for EXP-01).

For every character of every task sentence (true preceding text as context):
- bits per character over the LM alphabet (lower is better);
- top-1 / top-3 next-letter accuracy;
- latency of one prediction (median and p95, ms).
For every word boundary: top-3/5/10 next-word accuracy with an empty prefix (MA-30).

    python scripts/compare_fast_lm.py ngram kenlm:jonatasgrosman/wav2vec2-large-xlsr-53-russian
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eeg_speller.core.keyboard import LM_ALPHABET  # noqa: E402
from eeg_speller.text.normalize import normalize  # noqa: E402


def tasks() -> list[str]:
    out = []
    for line in (ROOT / "data" / "tasks_ru.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            out.append(normalize(line.split("\t", 1)[1]))
    return out


def load(spec: str):
    kind, _, arg = spec.partition(":")
    if kind == "ngram":
        from eeg_speller.llm.backends.char_ngram import CharNGram
        return CharNGram({"model_file": arg or "ru_char4.npz"})
    if kind.startswith("kenlm"):
        from eeg_speller.llm.backends.kenlm_words import KenLMWords, default_char_model, DEFAULT_REPO
        mix = float(kind.split("@")[1]) if "@" in kind else 0.1
        return KenLMWords({"model_id": arg or DEFAULT_REPO, "char_mix": mix}, default_char_model())
    if kind == "mock":
        from eeg_speller.core.config import load_config
        from eeg_speller.llm.backends.mock_ngram import MockNGram, _context

        class Mock(MockNGram):
            def lm_distribution(self, context):
                return self._dist(_context(context), tuple(LM_ALPHABET))
        return Mock(load_config(ROOT / "configs" / "default.yaml").fast_llm, ROOT / "data" / "corpus_ru_fallback.txt")
    raise SystemExit(f"unknown model {spec}")


def evaluate(model) -> dict:
    bits, top1, top3, n, lat = 0.0, 0, 0, 0, []
    word_hits = {3: 0, 5: 0, 10: 0}
    n_words = 0
    for text in tasks():
        for i, ch in enumerate(text):
            t = time.perf_counter()
            p = model.lm_distribution(text[:i])
            lat.append((time.perf_counter() - t) * 1000)
            k = LM_ALPHABET.index(ch)
            bits -= math.log2(max(float(p[k]), 1e-12))
            order = np.argsort(-p)
            top1 += int(order[0] == k)
            top3 += int(k in order[:3])
            n += 1
        for m in re.finditer(r"[а-я]+", text):
            if m.start() == 0 or text[m.start() - 1] != " ":
                continue
            cands = [w for w, _ in model.top_words(text[:m.start()], 10)]
            n_words += 1
            for k_ in word_hits:
                word_hits[k_] += int(m.group() in cands[:k_])
    return {"bits_per_char": round(bits / n, 3), "top1_letter": round(top1 / n, 3), "top3_letter": round(top3 / n, 3),
            "latency_ms_median": round(float(np.median(lat)), 2), "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
            **{f"top{k}_word": round(v / max(n_words, 1), 3) for k, v in word_hits.items()},
            "chars": n, "words": n_words}


if __name__ == "__main__":
    specs = sys.argv[1:] or ["ngram"]
    results = {}
    for spec in specs:
        t = time.time()
        model = load(spec)
        load_s = round(time.time() - t, 1)
        results[spec] = {**evaluate(model), "load_s": load_s}
        print(spec, json.dumps(results[spec], ensure_ascii=False), flush=True)
