"""Bank of real single-flash EEG responses, scored by the decoder, for any phrase.

A P300 speller decides from one number per flash: the decoder's log-likelihood ratio
"the attended symbol is in this flash" vs "it is not". This tool takes a recording with
known targets, calibrates the decoder on its first symbols and scores every flash of
the remaining (unseen) symbols. The result is two pools of real scores:

    target      flashes that contained the symbol the person was looking at
    nontarget   all other flashes

A stimulus module (stimulus_kit/modules/eeg_phrase) then "spells" any phrase by
drawing, for each flash, a real score from the matching pool. Real noise and real P300,
but on a phrase of your choice — with spaces, in any language (the P300 does not depend
on the letters). Not a full replay: the order of the responses is shuffled, so slow
changes within the session (fatigue) are lost.

    python stimulus_kit/tools/open_data/eeg_score_pool.py data/bci2_iib/bci2_iib_AAS.edf \
        --marker-map stimulus_kit/tools/open_data/markers_bci2_iib.json --train-epochs 42 \
        --out runs/models/bci2_iib_pool.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "stimulus_kit" / "decoders"))
sys.path.insert(0, str(ROOT / "stimulus_kit" / "tools"))


def build(edf, marker_map, train_epochs, pipeline, out):
    import external_pipeline_decoder as dec
    args = argparse.Namespace(edf=edf, marker_map=marker_map, train_epochs=train_epochs, pipeline=pipeline,
                              channels=None, out=str(Path(out).with_suffix(".joblib")), fit_scale=False)
    dec.calibrate(args)
    import joblib
    model = joblib.load(args.out)
    fs, names, epochs, data, mmap = dec.read_flashes(edf, marker_map, model["channels"])
    target, nontarget = [], []
    for ep in epochs[train_epochs:]:
        X, is_t = [], []
        for onset, syms in ep["flashes"]:
            x = dec.cut(data, fs, onset, model["cfg"])
            if x is not None:
                X.append(x)
                is_t.append(ep["cue"] in syms)
        s = dec.llr(model, np.array(X))
        is_t = np.array(is_t)
        target += list(s[is_t])
        nontarget += list(s[~is_t])
    target, nontarget = np.array(target, np.float32), np.array(nontarget, np.float32)
    d = (target.mean() - nontarget.mean()) / np.sqrt((target.var() + nontarget.var()) / 2)
    meta = {"recording": Path(edf).name, "pipeline": pipeline, "train_epochs": train_epochs,
            "test_symbols": len(epochs) - train_epochs, "n_target": int(len(target)),
            "n_nontarget": int(len(nontarget)), "d_prime": round(float(d), 3)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, target=target, nontarget=nontarget, meta=np.array(json.dumps(meta)))
    print(json.dumps(meta, ensure_ascii=False))
    return out


def load(path):
    with np.load(path) as z:
        return z["target"], z["nontarget"], json.loads(str(z["meta"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("edf")
    ap.add_argument("--marker-map", required=True)
    ap.add_argument("--train-epochs", type=int, required=True)
    ap.add_argument("--pipeline", default="xdawn_riemann")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.edf, a.marker_map, a.train_epochs, a.pipeline, a.out)


if __name__ == "__main__":
    main()
