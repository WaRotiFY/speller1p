"""Decoder that hosts a ready-made, third-party P300 pipeline behind the speller-link contract.

The speller only reads ``SpellerLikelihood``; what produces it is up to the
physiologists. This module shows how to plug in an existing open-source analyzer
instead of writing one: the "brain" is a standard pipeline from a public library,
this file only cuts EEG windows by the flash markers and turns the pipeline's
per-flash scores into symbol likelihoods.

Pipelines (``--pipeline``):

    xdawn_riemann  pyRiemann: XdawnCovariances → TangentSpace → LogisticRegression.
                   The reference P300 pipeline of the MOABB benchmark.
    rlda           scikit-learn: 0.5–10 Hz, 0–800 ms decimated to ~20 Hz, shrinkage LDA
                   (Krusienski et al. 2008; also what MEDUSA's CMDModelRLDA does).

Three commands:

    # 1. calibrate on the first N symbols of a labelled recording (cue = target letter)
    python stimulus_kit/decoders/external_pipeline_decoder.py calibrate rec.edf \
        --marker-map map.json --train-epochs 15 --pipeline xdawn_riemann --out model.joblib

    # 2. offline check on the remaining symbols: accuracy vs number of repetitions
    python stimulus_kit/decoders/external_pipeline_decoder.py evaluate rec.edf \
        --marker-map map.json --model model.joblib --skip-epochs 15

    # 3. online: SpellerEEG + SpellerStimuli -> SpellerLikelihood (same as p300_lsl_decoder.py)
    python stimulus_kit/decoders/external_pipeline_decoder.py run --model model.joblib --eeg-name SpellerEEG

The same ``preprocess`` runs offline and online, so the model sees identical input.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stimulus_kit"))
sys.path.insert(0, str(ROOT / "stimulus_kit" / "tools"))

PAD_S = 0.5                       # extra EEG before the window, so filters settle

PIPELINES = {
    "xdawn_riemann": {"band": (1.0, 20.0), "window": (0.0, 0.8), "decim_to": 64},
    "rlda": {"band": (0.5, 10.0), "window": (0.0, 0.8), "decim_to": 20},
}


def _flatten(x):
    return x.reshape(len(x), -1)


def make_pipeline(name):
    if name == "xdawn_riemann":
        from pyriemann.estimation import XdawnCovariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline as mk
        return mk(XdawnCovariances(nfilter=4, estimator="lwf", xdawn_estimator="scm"),
                  TangentSpace(metric="riemann"), LogisticRegression(C=1.0, max_iter=1000))
    if name == "rlda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import make_pipeline as mk
        from sklearn.preprocessing import FunctionTransformer
        return mk(FunctionTransformer(_flatten),
                  LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto"))
    raise SystemExit(f"unknown pipeline {name!r}; known: {', '.join(PIPELINES)}")


def preprocess(raw, fs, cfg):
    """raw: (channels, samples) from onset-PAD_S to onset+window[1] → (channels, t) ready for the pipeline."""
    sos = butter(4, cfg["band"], btype="bandpass", fs=fs, output="sos")
    x = sosfiltfilt(sos, raw, axis=-1)
    a = int(round((PAD_S + cfg["window"][0]) * fs))
    x = x[:, a:a + int(round((cfg["window"][1] - cfg["window"][0]) * fs))]
    step = max(1, int(round(fs / cfg["decim_to"])))
    return x[:, ::step]


def llr(model, X):
    """Per-flash log-likelihood ratio target vs non-target (training class prior removed)."""
    return model["pipeline"].decision_function(X) - model["prior_logit"]


# ───────────────────────────── offline: EDF + marker map ─────────────────────────────

def read_flashes(edf, marker_map, channels=None):
    """→ fs, channels, list of epochs: {"cue", "flashes": [(onset_s, symbols)]}, data (ch, n)."""
    from eeg_source import MarkerMap, load
    mmap = MarkerMap.load(marker_map)
    rec = load(str(edf), channels)
    data = rec.read(0, rec.n_samples).T.astype(float)
    eeg = [i for i, u in enumerate(rec.units) if u == "microvolts"]
    epochs = []
    for m in rec.markers:
        kind, val, _ = mmap.classify(m.label)
        if kind == "epoch_start" or (kind == "flash" and not epochs):
            epochs.append({"cue": None, "flashes": []})
        if kind == "cue":
            epochs[-1]["cue"] = val
        elif kind == "flash":
            epochs[-1]["flashes"].append((m.onset_s, val))
    names = [rec.ch_names[i] for i in eeg]
    rec.close()
    return rec.srate, names, epochs, data[eeg], mmap


def cut(data, fs, onset_s, cfg):
    a = int(round((onset_s - PAD_S + cfg["window"][0]) * fs))
    b = int(round((onset_s + cfg["window"][1]) * fs))
    if a < 0 or b > data.shape[1]:
        return None
    return preprocess(data[:, a:b], fs, cfg)


def calibrate(a):
    cfg = dict(PIPELINES[a.pipeline])
    fs, names, epochs, data, _ = read_flashes(a.edf, a.marker_map, a.channels)
    train = epochs[:a.train_epochs]
    X, y = [], []
    for ep in train:
        if ep["cue"] is None:
            raise SystemExit("calibration needs cue (target) markers in the recording")
        for onset, syms in ep["flashes"]:
            x = cut(data, fs, onset, cfg)
            if x is not None:
                X.append(x)
                y.append(int(ep["cue"] in syms))
    X, y = np.array(X), np.array(y)
    pipe = make_pipeline(a.pipeline).fit(X, y)
    p = y.mean()
    model = {"pipeline": pipe, "name": a.pipeline, "cfg": cfg, "channels": names, "fs": fs,
             "prior_logit": float(np.log(p / (1 - p))), "trained_on": str(a.edf),
             "train_epochs": len(train), "n_flashes": int(len(y))}
    import joblib
    joblib.dump(model, a.out)
    print(f"{a.pipeline}: trained on {len(train)} symbols, {len(y)} flashes ({y.sum()} target), "
          f"channels {names} → {a.out}")


def evaluate(a):
    import joblib
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    model = joblib.load(a.model)
    cfg = model["cfg"]
    fs, names, epochs, data, mmap = read_flashes(a.edf, a.marker_map, model["channels"])
    test = epochs[a.skip_epochs:]
    n_groups = sum(mmap.layout)                     # rows + columns = flashes per repetition
    max_rep = max(len(ep["flashes"]) for ep in test) // n_groups
    hits = np.zeros(max_rep)
    all_s, all_y, typed, wanted = [], [], "", ""
    for ep in test:
        X, groups = [], []
        for onset, syms in ep["flashes"]:
            x = cut(data, fs, onset, cfg)
            if x is not None:
                X.append(x)
                groups.append(syms)
        s = llr(model, np.array(X))
        all_s += list(s)
        all_y += [int(ep["cue"] in g) for g in groups]
        for r in range(1, max_rep + 1):
            logit = {sym: 0.0 for sym in mmap.symbols}
            for v, g in zip(s[:r * n_groups], groups[:r * n_groups]):
                for sym in g:
                    logit[sym] += v
            best = max(logit, key=logit.get)
            hits[r - 1] += best == ep["cue"]
            if r == max_rep:
                typed += best
        wanted += ep["cue"]
    all_s, all_y = np.array(all_s), np.array(all_y)
    report = {
        "edf": str(a.edf), "pipeline": model["name"], "test_symbols": len(test),
        "flash_auc": round(float(roc_auc_score(all_y, all_s)), 3),
        "flash_balanced_accuracy": round(float(balanced_accuracy_score(all_y, all_s > 0)), 3),
        "symbol_accuracy_by_repetitions": {r + 1: round(float(h / len(test)), 3) for r, h in enumerate(hits)},
        "wanted": wanted, "typed": typed,
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


# ───────────────────────────── online: LSL ─────────────────────────────

def _missed(source, t_start):
    """True if the window starts before the first EEG sample this decoder holds.

    Happens when the decoder connects to the EEG stream after the first flashes
    (e.g. a replay that another listener already started). Waiting would block
    the whole epoch, because flashes are scored strictly in order.
    """
    with source.lock:
        first = source.buffer[0][0][0] if source.buffer else None
    return first is not None and first > t_start

def run(a):
    import joblib
    from speller_link import SpellerLink
    from p300_lsl_decoder import FlashCollector        # same flash bookkeeping as the example decoder
    from eeg_speller.physiology.sources import LSLSource
    model = joblib.load(a.model)
    cfg = model["cfg"]
    link = SpellerLink(name=f"external_{model['name']}", role="decoder")
    collector = FlashCollector()
    print(f"pipeline {model['name']} · channels {model['channels']} · waiting for EEG {a.eeg_name!r} …", flush=True)
    source = LSLSource({"stream_field": "name", "stream_value": a.eeg_name, "units": "uV",
                        "connect_timeout_s": a.timeout, "marker_stream_name": "ExternalDecoderEcho"},
                       model["channels"])
    if abs(source.fs - model["fs"]) > 1e-6:
        print(f"warning: stream {source.fs:g} Hz, model trained at {model['fs']:g} Hz", flush=True)
    interval = (cfg["window"][0] - PAD_S, cfg["window"][1])
    state = {}
    while not link.ended:
        for eid in collector.epochs():
            st = state.setdefault(eid, {"scores": {}, "logits": None, "final": False})
            if st["final"]:
                continue
            if link.epoch_cancelled(eid):                     # speller-link@2: cancelled, nothing to send
                st["final"] = True
                continue
            flashes, done = collector.get(eid)
            if st["logits"] is None:
                syms = link.epoch.symbols if link.epoch else sorted({s for _, f in flashes for s in f})
                st["logits"] = {s: 0.0 for s in syms}
            changed = False
            for i, (onset, syms) in enumerate(flashes):
                if i in st["scores"]:
                    continue
                try:
                    w = source.window(onset, interval)
                except ValueError:
                    st["scores"][i] = None                     # gap inside the window
                    continue
                if w is None:
                    if _missed(source, onset + interval[0]):
                        st["scores"][i] = None                 # started before our EEG buffer: never coming
                        continue
                    break                                      # not here yet: keep evidence contiguous (through_flash)
                w = w * 1e6                                    # LSLSource gives volts; the model is in µV
                v = float(llr(model, preprocess(w, source.fs, cfg)[None])[0])
                st["scores"][i] = v
                for s in syms:
                    # a symbol first seen now has no evidence yet (decoder joined mid-epoch)
                    st["logits"][s] = st["logits"].get(s, 0.0) + v
                changed = True
            complete = done and len(st["scores"]) == len(flashes)
            if changed or complete:
                top = max(st["logits"].values())
                p = {s: float(np.exp(v - top)) for s, v in st["logits"].items()}
                ok = [v for v in st["scores"].values() if v is not None]
                link.send_likelihood(p, final=complete, epoch=eid, usable=bool(ok),
                                     through_flash=len(st["scores"]),
                                     info={"pipeline": model["name"], "flashes_scored": len(ok),
                                           "flashes_total": len(flashes),
                                           "llr_mean": round(float(np.mean(ok)), 3) if ok else None})
            st["final"] = complete
        time.sleep(0.005)
    source.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("edf")
    c.add_argument("--marker-map", required=True)
    c.add_argument("--train-epochs", type=int, default=15)
    c.add_argument("--pipeline", default="xdawn_riemann", choices=list(PIPELINES))
    c.add_argument("--channels", nargs="+")
    c.add_argument("--out", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("edf")
    e.add_argument("--marker-map", required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--skip-epochs", type=int, default=0, help="leave out the calibration symbols")
    e.add_argument("--json")
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--eeg-name", default="SpellerEEG")
    r.add_argument("--timeout", type=float, default=30.0)
    a = ap.parse_args(argv)
    {"calibrate": calibrate, "evaluate": evaluate, "run": run}[a.cmd](a)


if __name__ == "__main__":
    main()
