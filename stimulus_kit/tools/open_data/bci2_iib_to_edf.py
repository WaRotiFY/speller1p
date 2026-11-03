"""BCI Competition II, data set IIb (Wadsworth P300 speller, English words) -> EDF+ for the speller.

One participant, 64 channels, 240 Hz, the BCI2000 6×6 matrix (A–Z, 1–9, _), flash 100 ms,
SOA 175 ms, 15 repetitions of 6 columns + 6 rows per symbol (180 flashes).
Three sessions, one file per run (AAS0<session>R<run>.mat):

    10, 11  training, labelled (StimulusType):  cat dog fish water bowl | hat hat glove shoes fish rat
    12      test, unlabelled in the files:       food moot ham pie cake tuna zygot 4567

Labels of session 12 are the published answer of the competition (Blankertz et al. 2004,
IEEE TBME 51(6):1044, doi:10.1109/TBME.2004.826692). They match the run lengths
(4 4 3 3 4 4 5 4 symbols) and a decoder calibrated on sessions 10–11 spells them.
The words were spelled run by run: there are no spaces in the recording.

Units: the files hold A/D counts, the µV scale is not given. They are written as "uV"
so the speller pipeline treats them as EEG; only relative amplitudes are meaningful.

All runs go into one EDF+ in order 10, 11, 12 with annotations (markers_bci2_iib.json):

    epoch          start of one symbol
    target:x       the symbol spelled (lower case, like the speller)
    R1..R6, C1..C6 flash onsets (StimulusCode 7..12 = rows, 1..6 = columns)

    python stimulus_kit/tools/open_data/bci2_iib_to_edf.py AAS010R01.mat … --out data/bci2_iib
"""
import argparse
import os
import re
import sys

import numpy as np
import scipy.io as sio

MATRIX = list("abcdefghijklmnopqrstuvwxyz123456789_")   # BCI2000 6×6, row by row
FS = 240
PER_SYMBOL = 180                                       # 15 repetitions × 12 groups
TEST_TEXT = "foodmoothampiecaketunazygot4567"          # session 12, published answer
# 10–20 labels of the 64 channels (eloc64.txt of the competition, dots stripped)
CHANNELS = ("FC5 FC3 FC1 FCz FC2 FC4 FC6 C5 C3 C1 Cz C2 C4 C6 CP5 CP3 CP1 CPz CP2 CP4 CP6 Fp1 Fpz Fp2 "
            "AF7 AF3 AFz AF4 AF8 F7 F5 F3 F1 Fz F2 F4 F6 F8 FT7 FT8 T7 T8 T9 T10 TP7 TP8 P7 P5 P3 P1 Pz P2 "
            "P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2 Iz").split()


def read_run(path):
    d = sio.loadmat(path, squeeze_me=True)
    x = np.asarray(d["signal"], float)
    code = np.asarray(d["StimulusCode"], int)
    typ = np.asarray(d["StimulusType"], int) if "StimulusType" in d else None
    on = np.nonzero((np.diff(np.r_[0, code]) != 0) & (code > 0))[0]
    return x, code, typ, on


def run_symbols(code, typ, on):
    """Targets of a labelled run, one per 180 flashes."""
    out = []
    for b in range(0, len(on) // PER_SYMBOL * PER_SYMBOL, PER_SYMBOL):
        o = on[b:b + PER_SYMBOL]
        tg = sorted(set(code[o][typ[o] == 1]))
        if len(tg) != 2 or not (tg[0] <= 6 < tg[1]):
            sys.exit(f"cannot read the target of a symbol: codes {tg}")
        out.append(MATRIX[(tg[1] - 7) * 6 + (tg[0] - 1)])
    return out


def convert(mats, out_dir):
    import pyedflib
    mats = sorted(mats, key=lambda p: os.path.basename(p).upper())
    if not all(re.fullmatch(r"AAS01[012]R\d\d\.mat", os.path.basename(p), re.I) for p in mats):
        sys.exit("expected the competition files AAS010R01.mat … AAS012R08.mat")
    test_left = list(TEST_TEXT)
    xs, ann, text, offset = [], [], [], 0
    for path in mats:
        x, code, typ, on = read_run(path)
        n_sym = len(on) // PER_SYMBOL
        if typ is not None:
            syms = run_symbols(code, typ, on)
        else:
            syms, test_left = test_left[:n_sym], test_left[n_sym:]
        for k, sym in enumerate(syms):
            first = on[k * PER_SYMBOL]
            ann.append(((offset + first) / FS - 0.002, "epoch"))
            ann.append(((offset + first) / FS - 0.001, f"target:{sym}"))
            for i in on[k * PER_SYMBOL:(k + 1) * PER_SYMBOL]:
                c = code[i]
                ann.append(((offset + i) / FS, f"R{c - 6}" if c > 6 else f"C{c}"))
        text += syms
        xs.append(x)
        offset += len(x)
    if test_left:
        sys.exit(f"session 12 is incomplete: {len(test_left)} test symbols left over")
    X = np.concatenate(xs)
    ann.sort(key=lambda a: a[0])
    path = os.path.join(out_dir, "bci2_iib_AAS.edf")
    w = pyedflib.EdfWriter(path, X.shape[1], file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        hdrs = []
        for j, name in enumerate(CHANNELS):
            lim = float(np.ceil(np.max(np.abs(X[:, j])) * 1.05 + 1))
            hdrs.append({"label": name, "dimension": "uV", "sample_frequency": FS,
                         "physical_min": -lim, "physical_max": lim, "digital_min": -32768,
                         "digital_max": 32767, "transducer": "", "prefilter": "A/D counts"})
        w.setSignalHeaders(hdrs)
        w.setPatientCode("AAS")
        w.setEquipment("BCI2000")
        w.set_number_of_annotation_signals(8)
        for onset, label in ann:
            w.writeAnnotation(onset, -1, label)
        w.writeSamples([np.ascontiguousarray(X[:, j]) for j in range(X.shape[1])])
    finally:
        w.close()
    print(f"{path}: {len(text)} symbols, {sum(1 for _, a in ann if a[0] in 'RC')} flashes, text {''.join(text)!r}")
    return path, "".join(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mat", nargs="+", help="AAS010R01.mat … AAS012R08.mat (all 19)")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    convert(a.mat, a.out)


if __name__ == "__main__":
    main()
