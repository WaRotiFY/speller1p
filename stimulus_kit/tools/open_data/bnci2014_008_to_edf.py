"""BNCI 2014-008 (P300 speller, ALS patients) -> EDF+ that eeg_source.py can replay.

The dataset: 8 participants, 8 channels (Fz Cz Pz Oz P3 P4 PO7 PO8), 256 Hz, a 6×6
row/column matrix, 35 symbols per participant, 10 repetitions of 6 rows + 6 columns
per symbol (120 flashes), flash 125 ms, SOA 250 ms. Riccio et al. 2013,
doi:10.3389/fnhum.2013.00732; https://bnci-horizon-2020.eu/database/data-sets (008-2014).
License: CC BY-NC-ND 4.0.

Annotations written into the EDF+ (match markers_bnci2014_008.json):

    epoch          start of one symbol (the file's ``trial`` index)
    target:X       copy-spelling target of that symbol (calibration / scoring only)
    R1..R6, C1..C6 flash onsets (``y_stim`` codes 1..6 = rows, 7..12 = columns)

    python stimulus_kit/tools/open_data/bnci2014_008_to_edf.py A01.mat A02.mat … --out data/bnci2014_008
"""
import argparse
import os
import sys

import numpy as np
import scipy.io as sio

MATRIX = list("abcdefghijklmnopqrstuvwxyz123456789_")   # BCI2000 6×6, row by row; lower case like the speller
FS = 256


def read_mat(path):
    d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["data"]
    X = np.asarray(d.X, float)                    # (samples, 8), µV
    ys = np.asarray(d.y_stim, int)                # 0 or 1..12 while a group is lit
    y = np.asarray(d.y, int)                      # 0, 1 non-target, 2 target
    trials = list(np.asarray(d.trial, int) - 1)   # MATLAB 1-based -> 0-based
    return X, ys, y, trials, list(d.channels)


def flashes(ys):
    """Sample index and code of every flash onset."""
    on = np.nonzero((np.diff(np.r_[0, ys]) != 0) & (ys > 0))[0]
    return on, ys[on]


def targets(on, codes, y, trials, n):
    """Target symbol of each trial from its target flashes (≥ 2 of a code; one stray label is ignored)."""
    bounds = trials + [n]
    out = []
    for k in range(len(trials)):
        sel = (on >= bounds[k]) & (on < bounds[k + 1]) & (y[on] == 2)
        cnt = np.bincount(codes[sel], minlength=13)[1:]
        r, c = int(np.argmax(cnt[:6])), int(np.argmax(cnt[6:]))
        if cnt[r] < 2 or cnt[6 + c] < 2:
            sys.exit(f"trial {k}: cannot find target row/column, counts {cnt}")
        out.append(MATRIX[r * 6 + c])
    return out


def write_edf(path, X, channels, annotations, info):
    import pyedflib
    n_ch = X.shape[1]
    w = pyedflib.EdfWriter(path, n_ch, file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        hdrs = []
        for j, name in enumerate(channels):
            lim = float(np.ceil(np.max(np.abs(X[:, j])) * 1.05 + 1))
            hdrs.append({"label": name, "dimension": "uV", "sample_frequency": FS,
                         "physical_min": -lim, "physical_max": lim,
                         "digital_min": -32768, "digital_max": 32767, "transducer": "", "prefilter": ""})
        w.setSignalHeaders(hdrs)
        w.setPatientCode(info["subject"])
        w.setEquipment("g.USBamp")
        # ~4 flashes/s: give the annotation channel room so no marker is lost
        w.set_number_of_annotation_signals(8)
        for onset, text in annotations:
            w.writeAnnotation(onset, -1, text)
        w.writeSamples([np.ascontiguousarray(X[:, j]) for j in range(n_ch)])
    finally:
        w.close()


def convert(mat, out_dir):
    X, ys, y, trials, channels = read_mat(mat)
    on, codes = flashes(ys)
    tgt = targets(on, codes, y, trials, len(ys))
    ann = []
    for k, t0 in enumerate(trials):
        # strictly before the first flash of the symbol (it may start at sample t0)
        ann.append((t0 / FS - 0.002, "epoch"))
        ann.append((t0 / FS - 0.001, f"target:{tgt[k]}"))
    for i, c in zip(on, codes):
        ann.append((i / FS, f"R{c}" if c <= 6 else f"C{c - 6}"))
    ann.sort(key=lambda a: a[0])
    subject = os.path.splitext(os.path.basename(mat))[0]
    path = os.path.join(out_dir, f"bnci2014_008_{subject}.edf")
    write_edf(path, X, channels, ann, {"subject": subject})
    print(f"{path}: {len(trials)} symbols, {len(on)} flashes, text {''.join(tgt)!r}")
    return path, "".join(tgt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mat", nargs="+", help="A01.mat … A08.mat")
    ap.add_argument("--out", default=".", help="output folder")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for m in a.mat:
        convert(m, a.out)


if __name__ == "__main__":
    main()
