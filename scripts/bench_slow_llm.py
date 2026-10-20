"""Benchmark a slow (correcting) LLM on realistic speller errors.

1. ``make``: types every phrase of data/tasks_ru.txt with a simulated decoder (noisy P300
   likelihood, quality d'), the fast LLM and the operator-window controller, and records
   every correction request exactly as the speller would send it (window, left context,
   letter alternatives) together with the true words.
2. ``run``: sends those requests to a chat API model and counts what it fixed and broke.

    python scripts/bench_slow_llm.py make --d-prime 1.6 --out runs/bench_slow/cases.json
    python scripts/bench_slow_llm.py run runs/bench_slow/cases.json --provider ollama --model qwen2.5:7b
    python scripts/bench_slow_llm.py run runs/bench_slow/cases.json --model qwen2.5:7b --prompt prompts/correction_v2.txt

Metrics per model: windows with an error that came back fully correct; erroneous words
fixed; correct words broken; replies rejected (wrong word count, empty reply, error);
median latency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eeg_speller.external.controller import ExternalSession, default_model_factory  # noqa: E402
from eeg_speller.external.server import build_session_config  # noqa: E402
from eeg_speller.external.tasks import read_task_list  # noqa: E402
from eeg_speller.external.transport import MemoryTransport  # noqa: E402
from eeg_speller.core.keyboard import typed_char  # noqa: E402
from eeg_speller.llm.backends import chat_api  # noqa: E402
from eeg_speller.llm.slow import align_correction, validate_correction  # noqa: E402

WORD = re.compile(r"[а-я]+")


class Recorder:
    """Slow-LLM stand-in that records the requests and changes nothing."""
    version = "chat-api@recorder"
    uses_letters = True
    model_id = "recorder"

    def __init__(self):
        self.requests = []

    def correct(self, left, words, letters=None):
        self.requests.append({"left": left, "words": list(words), "letters": letters})
        return list(words)


def simulate(target: str, d_prime: float, rng, out_dir: Path):
    cfg, _ = build_session_config({"slow": {"backend": "chat_api", "provider": "ollama", "model_id": "x",
                                            "every_n_words": 1, "window_words": 3, "alternatives_k": 4}})
    cfg = cfg.model_copy(update={"run_mode": "live", "stimulus": {**cfg.stimulus, "pause_ms": 0},
                                 "slow_llm": {**cfg.slow_llm, "min_context_words": 0}})
    rec = Recorder()
    tr = MemoryTransport()

    def factory(kind, c):
        return rec if kind == "slow" else default_model_factory(kind, c)

    def module(msg):
        if msg["cmd"] != "epoch_start":
            return
        k = len(session.primary.value)
        want = target[min(k, len(target) - 1)]
        symbols = msg["symbols"]
        # Row/column P300 evidence: every symbol gets a noisy score, the intended one a shift of d'.
        score = rng.normal(0.0, 1.0, len(symbols))
        score[[typed_char(s) for s in symbols].index(want)] += d_prime * 2.2
        p = np.exp(score * d_prime)
        tr.deliver("likelihood", {"epoch": msg["epoch"], "p": dict(zip(symbols, (p / p.sum()).tolist())),
                                  "final": True})

    tr.listeners.append(module)
    session = ExternalSession(cfg, target, out_dir, transport=tr, model_factory=factory)
    session.start()
    session.wait_finished(600)
    return session.primary.value, rec.requests


def true_words_for(request, typed: str, target: str):
    """True words of a window, or None when they cannot be paired one to one.

    The simulated participant always aims at target[len(typed)], so typed character i is an
    attempt at target character i. A window is kept only when every typed word covers
    exactly one task word (no space typed in the wrong place) and the text before it has
    not changed since the request.
    """
    left = request["left"]
    if not typed.startswith(left):
        return None
    truth, pos = [], len(left)
    for word in request["words"]:
        m = WORD.search(typed, pos)
        if m is None or m.group() != word:
            return None
        a, b = m.span()
        span = target[a:b]
        inside = WORD.fullmatch(span) is not None
        before_ok = a == 0 or not target[a - 1].isalpha()
        after_ok = b >= len(target) or not target[b].isalpha()
        if not (inside and before_ok and after_ok):
            return None
        truth.append(span)
        pos = b
    return truth


def make(args):
    rng = np.random.default_rng(args.seed)
    cases = []
    tasks = read_task_list()
    for t in tasks[: args.limit]:
        target = t["text"]
        typed, requests = simulate(target, args.d_prime, rng, Path(args.out).parent / "sessions")
        for r in requests:
            truth = true_words_for(r, typed, target)
            if truth is not None:
                cases.append({"task": t["id"], **r, "truth": truth})
        print(f"{t['id']}: {typed!r}", file=sys.stderr)
    # Keep each distinct window once; the last window of a phrase is the most complete one.
    seen, unique = set(), []
    for c in cases:
        key = (c["left"], tuple(c["words"]))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(unique, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = sum(c["words"] != c["truth"] for c in unique)
    print(json.dumps({"cases": len(unique), "with_errors": bad,
                      "wrong_words": sum(a != b for c in unique for a, b in zip(c["words"], c["truth"]))}))


def run(args):
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    if args.only_errors:
        cases = [c for c in cases if c["words"] != c["truth"]]
    cases = cases[: args.limit]
    prompt = Path(args.prompt).read_text(encoding="utf-8") if args.prompt else None
    model = chat_api.ChatCorrection({"provider": args.provider, "base_url": args.base_url, "model_id": args.model,
                                     "timeout_s": args.timeout, "max_tokens": args.max_tokens,
                                     "prompt_text": prompt}, allow_remote=True)
    stats = {"windows": 0, "windows_with_errors": 0, "windows_fixed": 0, "wrong_words": 0, "words_fixed": 0,
             "right_words": 0, "words_broken": 0, "rejected": 0}
    latency, examples = [], []
    for c in cases:
        start = time.perf_counter()
        try:
            model.last_reply = None
            out = model.correct(c["left"], c["words"], c["letters"] if not args.no_alternatives else None)
            if args.strict:
                ok, why = validate_correction(c["words"], out, True)
            else:
                out, ok, why = align_correction(c["words"], out)      # what the speller does by default
        except ValueError as exc:
            out, ok, why = list(c["words"]), False, str(exc)[:80]
        latency.append(time.perf_counter() - start)
        if not ok:
            stats["rejected"] += 1
            out = list(c["words"])           # a rejected reply leaves the text as typed
        stats["windows"] += 1
        has_err = c["words"] != c["truth"]
        stats["windows_with_errors"] += has_err
        stats["windows_fixed"] += has_err and out == c["truth"]
        for typed, true, got in zip(c["words"], c["truth"], out):
            if typed != true:
                stats["wrong_words"] += 1
                stats["words_fixed"] += got == true
            else:
                stats["right_words"] += 1
                stats["words_broken"] += got != true
        examples.append({"in": " ".join(c["words"]), "out": " ".join(out), "truth": " ".join(c["truth"]),
                         "ok": ok, "why": why, "reply": model.last_reply})
        if args.verbose:
            mark = "✓" if out == c["truth"] else ("·" if out == c["words"] else "✗")
            print(f"{mark} {' '.join(c['words']):30} → {' '.join(out):30} [{' '.join(c['truth'])}]"
                  + ("" if ok else f"  ({why}; reply {model.last_reply!r})"), file=sys.stderr, flush=True)
    stats["median_latency_s"] = round(statistics.median(latency), 2) if latency else None
    result = {"model": args.model, "provider": args.provider, "prompt": args.prompt or "current", **stats}
    print(json.dumps(result, ensure_ascii=False))
    if args.save:
        Path(args.save).write_text(json.dumps({"result": result, "examples": examples}, ensure_ascii=False,
                                              indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--d-prime", type=float, default=1.6, help="signal quality of the simulated decoder")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--limit", type=int, default=40, help="how many task phrases to type")
    m.add_argument("--out", default="runs/bench_slow/cases.json")
    r = sub.add_parser("run")
    r.add_argument("cases")
    r.add_argument("--provider", default="ollama")
    r.add_argument("--base-url", default=None)
    r.add_argument("--model", required=True)
    r.add_argument("--timeout", type=float, default=120)
    r.add_argument("--max-tokens", type=int, default=None)
    r.add_argument("--prompt", help="prompt file (default: the current speller prompt)")
    r.add_argument("--limit", type=int, default=1000)
    r.add_argument("--only-errors", action="store_true", help="only windows that contain an error")
    r.add_argument("--no-alternatives", action="store_true", help="do not send letter alternatives")
    r.add_argument("--strict", action="store_true", help="reject replies with another word count (old rule)")
    r.add_argument("--save", help="write the per-window results here")
    r.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    make(args) if args.cmd == "make" else run(args)


if __name__ == "__main__":
    main()
