"""Open-data path: BNCI 2014-008 .mat -> EDF+ -> marker map -> epochs the replay module plays."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
OPEN = ROOT / "stimulus_kit" / "tools" / "open_data"
sys.path.insert(0, str(OPEN))
sys.path.insert(0, str(ROOT / "stimulus_kit" / "tools"))
sys.path.insert(0, str(ROOT / "stimulus_kit"))
pytest.importorskip("pyedflib")


def fake_bnci(path, targets=((0, 2), (3, 1)), reps=2, fs=256):
    """Two symbols in the BNCI 2014-008 layout: y_stim 1..6 rows, 7..12 columns, y 2 = target."""
    flash, soa, gap = 32, 64, 400
    n = gap + len(targets) * (reps * 12 * soa + gap)
    ys = np.zeros(n, np.int16)
    y = np.zeros(n, np.uint8)
    trial = []
    i = gap
    rng = np.random.default_rng(0)
    for row, col in targets:
        trial.append(i + 1)                       # MATLAB 1-based
        for _ in range(reps):
            for code in rng.permutation(np.arange(1, 13)):
                ys[i:i + flash] = code
                y[i:i + flash] = 2 if code in (row + 1, col + 7) else 1
                i += soa
        i += gap
    X = rng.normal(0, 10, (n, 8))
    data = {"X": X, "y": y, "y_stim": ys, "trial": np.array(trial, np.int32),
            "channels": np.array(["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"], dtype=object),
            "classes": np.array(["NonTarget", "Target"], dtype=object)}
    sio.savemat(path, {"data": data})


def test_bnci_mat_to_edf_and_back_to_epochs(tmp_path):
    from bnci2014_008_to_edf import convert
    from eeg_source import MarkerMap, load
    fake_bnci(tmp_path / "A99.mat")
    edf, text = convert(str(tmp_path / "A99.mat"), str(tmp_path))
    assert text == "ct"                            # row 0/col 2 -> c; row 3/col 1 -> t (lower case, like the speller)

    spec = importlib.util.spec_from_file_location("eeg_replay_main", ROOT / "stimulus_kit/modules/eeg_replay/main.py")
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    mmap = MarkerMap.load(str(OPEN / "markers_bnci2014_008.json"))
    rec = load(edf)
    epochs = replay.read_epochs(rec, mmap)
    assert [e["cue"] for e in epochs] == ["c", "t"]
    assert [len(e["flashes"]) for e in epochs] == [24, 24]
    for e in epochs:                               # the target is in exactly 2 flashes per repetition
        assert sum(e["cue"] in syms for _, syms in e["flashes"]) == 4
        assert all(b[0] - a[0] == pytest.approx(0.25, abs=1e-3) for a, b in zip(e["flashes"], e["flashes"][1:]))
    assert rec.ch_names == ["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"] and rec.srate == 256
    rec.close()


def test_marker_map_matches_bnci_matrix():
    from eeg_source import MarkerMap
    from bnci2014_008_to_edf import MATRIX
    mmap = MarkerMap.load(str(OPEN / "markers_bnci2014_008.json"))
    assert mmap.symbols == MATRIX and mmap.layout == (6, 6)
    assert mmap.classify("R2") == ("flash", list("ghijkl"), None)
    assert mmap.classify("C6") == ("flash", list("flrx4_"), None)
    assert mmap.classify("target:g") == ("cue", "g", None)
