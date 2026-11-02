"""Scores a replay: final SpellerLikelihood of each epoch vs the target letter in the recording.

Run next to ``eeg_source.py replay … --as-stimulus --as-control`` and a decoder. The
k-th ``target:X`` marker in SpellerReplayMarkers is the target of epoch k of the
replay (eeg_source numbers epochs from 0 in playing order). Targets never reach the
decoder: they travel only in SpellerReplayMarkers.

    python stimulus_kit/tools/open_data/replay_score.py --cue "^target:(.+)$" --out score.json
"""
import argparse
import json
import re
import sys
import time

import pylsl


def inlet(name, timeout):
    found = pylsl.resolve_byprop("name", name, 1, timeout)
    if not found:
        sys.exit(f"no LSL stream {name!r}")
    i = pylsl.StreamInlet(found[0], processing_flags=pylsl.proc_clocksync)
    i.open_stream(timeout=5)
    return i


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cue", default=r"^target:(.+)$", help="regex; group 1 is the target letter")
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--out")
    a = ap.parse_args()
    cue_re = re.compile(a.cue)
    lik = inlet("SpellerLikelihood", a.timeout)
    marks = inlet("SpellerReplayMarkers", a.timeout)
    ctrl = inlet("SpellerControl", a.timeout)
    cues, finals, first_update, ended = [], {}, {}, False
    print("scoring…", flush=True)
    last = time.monotonic()
    while not ended or time.monotonic() - last < 3:
        got = False
        for s, ts in zip(*marks.pull_chunk(timeout=0.0)):
            m = cue_re.match(json.loads(s[0])["label"])
            if m:
                cues.append(m.group(1))
            got = True
        for s, ts in zip(*lik.pull_chunk(timeout=0.0)):
            msg = json.loads(s[0])
            first_update.setdefault(msg["epoch"], ts)
            if msg.get("final"):
                p = msg["p"]
                best = max(p, key=p.get)
                total = sum(p.values())
                finals[msg["epoch"]] = (best, p[best] / total, msg.get("info", {}))
                k = msg["epoch"]
                want = cues[k] if k < len(cues) else "?"
                print(f"epoch {k:3d}: typed {best!r} (p={p[best] / total:.2f}) want {want!r} "
                      f"{'ok' if best == want else 'MISS'} {msg.get('info', {})}", flush=True)
            got = True
        for s, ts in zip(*ctrl.pull_chunk(timeout=0.0)):
            if json.loads(s[0]).get("cmd") == "end":
                ended = True
        if got:
            last = time.monotonic()
        if time.monotonic() - last > a.timeout:
            break
        time.sleep(0.01)
    ks = sorted(finals)
    typed = "".join(finals[k][0] for k in ks)
    wanted = "".join(cues[k] if k < len(cues) else "?" for k in ks)
    acc = sum(t == w for t, w in zip(typed, wanted)) / max(1, len(ks))
    report = {"epochs": len(ks), "typed": typed, "wanted": wanted, "symbol_accuracy": round(acc, 3)}
    print(json.dumps(report, ensure_ascii=False))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
