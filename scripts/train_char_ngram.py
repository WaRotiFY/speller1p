"""Train the default Russian character n-gram (interpolated Kneser-Ney) for the fast LM slot.

The result is a lookup table P(next char | previous n-1 chars) over the LM alphabet
(32 letters with ё merged into е, space, period, comma) plus a word list for
next-word candidates. It is saved as one compact ``.npz`` that ``ngram`` loads.

Sources used for the shipped model (see models/ngram/README.md):

    python scripts/train_char_ngram.py \
        --lenta lenta-ru-news.csv.bz2 \
        --texts RussianNovels/corpus \
        --wordfreq FrequencyWords/content/2018/ru/ru_full.txt \
        --order 4 --out models/ngram/ru_char4.npz

Any folder of UTF-8 ``.txt`` files works with ``--texts``. Sentences from
``data/tasks_ru.txt`` are removed from the training text.
"""
from __future__ import annotations

import argparse
import bz2
import csv
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

ALPHABET = "абвгдежзийклмнопрстуфхцчшщъыьэюя .,"
SPACE = ALPHABET.index(" ")
V = len(ALPHABET)
ROOT = Path(__file__).resolve().parents[1]

_table = {ord(c): c for c in ALPHABET}
_table.update({ord(c.upper()): c for c in ALPHABET if c.isalpha()})
_table.update({ord("ё"): "е", ord("Ё"): "е"})
for ch in "!?;…":
    _table[ord(ch)] = "."
for ch in "\n\t\r -—–:()\"«»'/":
    _table[ord(ch)] = " "


_drop = re.compile(r"[^а-я .,]")


def normalize(text: str) -> str:
    """Map raw text to the LM alphabet; everything else is dropped."""
    text = re.sub(r"[0-9]+([.,][0-9]+)*", " ", text)
    out = _drop.sub("", text.translate(_table))
    out = re.sub(r" *([.,])[ .,]*", r"\1 ", out)
    return re.sub(r" {2,}", " ", out)


def encode(text: str) -> np.ndarray:
    lut = np.full(0x500, 255, dtype=np.uint8)
    for i, c in enumerate(ALPHABET):
        lut[ord(c)] = i
    codes = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    codes = np.minimum(codes, 0x4FF)
    out = lut[codes]
    return out[out != 255]


class Counter:
    def __init__(self, order: int):
        self.order = order
        self.counts = np.zeros(V ** order, dtype=np.int64)
        self.chars = 0

    def _index(self, ids: np.ndarray) -> np.ndarray:
        idx = np.zeros(len(ids) - self.order + 1, dtype=np.int64)
        for k in range(self.order):
            idx = idx * V + ids[k:len(ids) - self.order + 1 + k]
        return idx

    def add(self, text: str):
        # Every document starts after (order-1) spaces, so short contexts are well defined.
        ids = encode(" " * (self.order - 1) + text)
        if len(ids) < self.order:
            return
        self.counts += np.bincount(self._index(ids), minlength=V ** self.order)
        self.chars += len(ids)

    def add_weighted(self, items: list[tuple[str, int]]):
        """Many short texts with integer weights (e.g. a word frequency list), vectorized."""
        for start in range(0, len(items), 50000):
            idx_parts, w_parts = [], []
            for text, weight in items[start:start + 50000]:
                ids = encode(" " * (self.order - 1) + text)
                if len(ids) < self.order:
                    continue
                idx = self._index(ids)
                idx_parts.append(idx)
                w_parts.append(np.full(len(idx), weight, dtype=np.float64))
                self.chars += len(ids) * weight
            if idx_parts:
                self.counts += np.round(np.bincount(np.concatenate(idx_parts), weights=np.concatenate(w_parts),
                                                    minlength=V ** self.order)).astype(np.int64)


def discount(counts: np.ndarray) -> float:
    n1, n2 = int((counts == 1).sum()), int((counts == 2).sum())
    return n1 / (n1 + 2 * n2) if n1 + 2 * n2 else 0.5


def kneser_ney(counts: np.ndarray, order: int) -> np.ndarray:
    """Interpolated KN: returns P[context_index, next_char] for (order-1)-char contexts."""
    shape = (V,) * order
    c = counts.reshape(shape).astype(np.float64)
    # Lower orders use continuation counts N1+(. w), as in Kneser-Ney.
    levels = [c]
    for _ in range(order - 1):
        levels.append((levels[-1] > 0).sum(axis=0).astype(np.float64))
    levels.reverse()                 # levels[k] has k+1 axes: context of k chars + next char
    p = None
    for k, table in enumerate(levels):
        if k == 0:
            total = table.sum()
            p = (table + 1) / (total + V)              # add-one at the unigram level
            continue
        d = discount(table[table > 0].ravel())
        ctx_total = table.sum(axis=-1, keepdims=True)
        n_types = (table > 0).sum(axis=-1, keepdims=True)
        lower = np.broadcast_to(p, table.shape)       # p has k axes: last k-1 context chars + next
        with np.errstate(invalid="ignore", divide="ignore"):
            prob = np.maximum(table - d, 0) / ctx_total + d * n_types / ctx_total * lower
        p = np.where(ctx_total > 0, prob, lower)
    return p.reshape(V ** (order - 1), V).astype(np.float32)


def bits_per_char(table: np.ndarray, order: int, text: str) -> float:
    ids = encode(" " * (order - 1) + normalize(text))
    total, n = 0.0, 0
    for i in range(order - 1, len(ids)):
        ctx = 0
        for k in range(order - 1):
            ctx = ctx * V + int(ids[i - order + 1 + k])
        total -= np.log2(float(table[ctx, int(ids[i])]))
        n += 1
    return total / max(n, 1)


def task_sentences() -> set[str]:
    out = set()
    for line in (ROOT / "data" / "tasks_ru.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "\t" in line:
            out.add(normalize(line.split("\t", 1)[1]).strip(" ."))
    return out


def drop_tasks(text: str, tasks: set[str]) -> str:
    parts = re.split(r"(?<=\.) ", text)
    return " ".join(p for p in parts if p.strip(" .") not in tasks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lenta", type=Path, help="lenta-ru-news.csv(.bz2)")
    ap.add_argument("--lenta-max-mb", type=float, default=0, help="stop after this many MB of text (0 = all)")
    ap.add_argument("--texts", type=Path, action="append", default=[], help="folder of .txt files")
    ap.add_argument("--wordfreq", type=Path, help="'word count' list (e.g. OpenSubtitles frequency list)")
    ap.add_argument("--wordfreq-share", type=float, default=0.15,
                    help="share of all characters contributed by the frequency list")
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=40000, help="words kept for next-word candidates")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "ngram" / "ru_char4.npz")
    args = ap.parse_args()

    tasks = task_sentences()
    counter = Counter(args.order)
    words: dict[str, float] = {}
    sources = []

    def add_doc(text: str):
        clean = drop_tasks(normalize(text), tasks)
        if len(clean) < 20:
            return
        counter.add(clean)
        for w in re.findall(r"[а-я]+", clean):
            words[w] = words.get(w, 0) + 1

    t0 = time.time()
    for folder in args.texts:
        files = sorted(folder.glob("**/*.txt"))
        for f in files:
            add_doc(f.read_text(encoding="utf-8", errors="ignore"))
        sources.append({"kind": "texts", "path": folder.name, "files": len(files)})
        print(f"texts {folder}: {len(files)} files, {counter.chars / 1e6:.0f}M chars, {time.time() - t0:.0f}s",
              flush=True)
    if args.lenta:
        opener = bz2.open if args.lenta.suffix == ".bz2" else open
        csv.field_size_limit(sys.maxsize)
        start, n = counter.chars, 0
        with opener(args.lenta, "rt", encoding="utf-8", newline="") as fh:
            batch = []
            for row in csv.DictReader(fh):
                batch.append(row.get("text") or "")
                if len(batch) >= 2000:
                    add_doc(" ".join(batch))
                    n += len(batch)
                    batch = []
                    if n % 100000 == 0:
                        print(f"lenta: {n} articles, {(counter.chars - start) / 1e6:.0f}M chars, "
                              f"{time.time() - t0:.0f}s", flush=True)
                    if args.lenta_max_mb and counter.chars - start > args.lenta_max_mb * 1e6:
                        break
            if batch:
                add_doc(" ".join(batch))
                n += len(batch)
        sources.append({"kind": "lenta", "path": args.lenta.name, "articles": n})
    if args.wordfreq:
        rows = []
        for line in args.wordfreq.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                w = normalize(parts[0]).strip(" .,")
                if re.fullmatch(r"[а-я]+", w or ""):
                    rows.append((w, int(parts[1])))
        total = sum(c * (len(w) + 1) for w, c in rows)
        target = counter.chars * args.wordfreq_share / (1 - args.wordfreq_share)
        scale = target / total if total else 0
        # Words are added as " word " with integer weights; rare words get at least 1.
        counter.add_weighted([(w + " ", max(1, round(c * scale))) for w, c in rows])
        for w, c in rows:
            words[w] = words.get(w, 0) + c * scale
        sources.append({"kind": "wordfreq", "path": args.wordfreq.name, "words": len(rows),
                        "share": args.wordfreq_share})
        print(f"wordfreq: {len(rows)} words, scale {scale:.4f}", flush=True)

    print(f"counting done: {counter.chars / 1e6:.0f}M chars, {time.time() - t0:.0f}s; smoothing", flush=True)
    table = kneser_ney(counter.counts, args.order)
    top = sorted(words.items(), key=lambda kv: -kv[1])[:args.vocab]
    task_text = " ".join(sorted(tasks))
    bpc = bits_per_char(table, args.order, task_text)
    meta = {"format": "eeg-speller/char-ngram@1", "alphabet": ALPHABET, "order": args.order,
            "smoothing": "interpolated Kneser-Ney (one discount per order, add-one unigram)",
            "train_chars": int(counter.chars), "sources": sources,
            "bits_per_char_tasks": round(float(bpc), 3), "vocab_size": len(top),
            "created": time.strftime("%Y-%m-%d")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, table=table.astype(np.float16),
                        vocab=np.array([w for w, _ in top]),
                        vocab_count=np.array([c for _, c in top], dtype=np.float32),
                        meta=np.array(json.dumps(meta, ensure_ascii=False)))
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
