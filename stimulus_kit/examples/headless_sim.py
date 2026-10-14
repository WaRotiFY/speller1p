"""Без экрана: симуляция вспышек и «мозга» для проверки.

Useful for testing the speller and the protocol on any computer (CI, servers).
The simulated participant knows the phrase it wants to type (``--text``) —
that knowledge lives here, in the module, never in the speller.

    python stimulus_kit/examples/headless_sim.py --text "привет" --d-prime 1.0
"""
import argparse
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from speller_link import SpellerLink, now  # noqa: E402


def run(text, d_prime=1.0, flash_ms=None, isi_ms=None, repetitions=None, seed=0,
        name="headless_sim", wait_for_speller=30.0, link=None, verbose=True):
    rng = random.Random(seed)
    link = link or SpellerLink(name=name, role="stimulus+decoder", wait_for_speller=wait_for_speller)
    flashes_sent = 0
    while True:
        epoch = link.wait_epoch(timeout=60)
        if epoch is None:
            break
        cfg = epoch.settings
        flash_s = (flash_ms or cfg.get("flash_ms", 100)) / 1000
        soa = flash_s + (isi_ms if isi_ms is not None else cfg.get("isi_ms", 150)) / 1000
        reps = repetitions or cfg.get("n_repetitions", 5)
        pos = len(epoch.text)
        target = text[pos] if pos < len(text) else " "
        rows = epoch.rows()
        groups = list(rows) + [[r[c] for r in rows] for c in range(len(rows[0]))]
        logl = {s: 0.0 for s in epoch.symbols}
        t = now() + 0.05
        for _ in range(reps):
            order = groups[:]
            rng.shuffle(order)
            for group in order:
                while now() < t:
                    time.sleep(0.0005)
                link.flash(group, timestamp=t)
                flashes_sent += 1
                # Gaussian classifier score: N(d', 1) for the attended group, N(0, 1) otherwise.
                x = rng.gauss(d_prime if target in group else 0.0, 1.0)
                llr = d_prime * x - d_prime ** 2 / 2
                for s in group:
                    logl[s] += llr
                link.send_likelihood(_softmax(logl))
                t += soa
                if link.should_stop():
                    break
            if link.should_stop():
                break
        link.done()
        link.send_likelihood(_softmax(logl), final=True)
        res = link.wait_result(timeout=10)
        if verbose and res:
            print("epoch %d: chose %r, text %r" % (epoch.id, res.get("symbol"), res.get("text")), flush=True)
    return flashes_sent


def _softmax(logl):
    top = max(logl.values())
    w = {s: math.exp(v - top) for s, v in logl.items()}
    total = sum(w.values())
    return {s: v / total for s, v in w.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", required=True, help="phrase the simulated participant wants to type")
    ap.add_argument("--d-prime", type=float, default=1.0, help="signal quality of the simulated brain")
    ap.add_argument("--flash-ms", type=int)
    ap.add_argument("--isi-ms", type=int)
    ap.add_argument("--repetitions", type=int)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.text.lower().replace("ё", "е"), args.d_prime, args.flash_ms, args.isi_ms, args.repetitions, args.seed)


if __name__ == "__main__":
    main()
