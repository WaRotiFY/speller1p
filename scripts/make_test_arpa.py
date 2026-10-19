"""Write a small but valid word bigram ARPA model (absolute discounting with backoff).

Only for tests and demos of the word-KenLM fast model without a 1.5 GB download:

    python scripts/make_test_arpa.py data/corpus_ru_fallback.txt out.arpa
"""
from collections import Counter, defaultdict
import math
import re
import sys


def build(text: str, discount: float = 0.7):
    sentences = [re.findall(r"[а-яё]+", s.lower()) for s in re.split(r"[.!?\n]+", text)]
    sentences = [["<s>"] + s + ["</s>"] for s in sentences if s]
    uni = Counter(w for s in sentences for w in s if w != "<s>")
    bi = defaultdict(Counter)
    for s in sentences:
        for a, b in zip(s, s[1:]):
            bi[a][b] += 1
    total = sum(uni.values())
    vocab = sorted(uni) + ["<unk>"]
    p1 = {w: (uni[w] + 0.5) / (total + 0.5 * len(vocab)) for w in vocab}
    p1["<s>"] = 0.0
    lines2, backoff = [], {}
    for a, nexts in bi.items():
        ca = sum(nexts.values())
        seen = 0.0
        for b, c in nexts.items():
            p = (c - discount) / ca
            lines2.append((a, b, p))
            seen += p1[b]
        left = discount * len(nexts) / ca
        backoff[a] = left / max(1e-9, 1 - seen)
    out = ["", "\\data\\", f"ngram 1={len(vocab) + 1}", f"ngram 2={len(lines2)}", "", "\\1-grams:"]
    for w in ["<s>"] + vocab:
        lp = -99.0 if w == "<s>" else math.log10(p1[w])
        bo = backoff.get(w)
        out.append(f"{lp:.6f}\t{w}" + (f"\t{math.log10(bo):.6f}" if bo else ""))
    out += ["", "\\2-grams:"]
    out += [f"{math.log10(p):.6f}\t{a} {b}" for a, b, p in lines2]
    out += ["", "\\end\\", ""]
    return "\n".join(out)


if __name__ == "__main__":
    with open(sys.argv[2], "w", encoding="utf-8") as fh:
        fh.write(build(open(sys.argv[1], encoding="utf-8").read()))
