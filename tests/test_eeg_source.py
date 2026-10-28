"""EEG source tool (stimulus_kit/tools/eeg_source.py): marker maps, EDF reading and real LSL replay."""
import json
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stimulus_kit" / "tools"))
sys.path.insert(0, str(ROOT / "stimulus_kit"))

FS = 256
ROWS = [["а", "б", "в"], ["г", "д", "е"]]
SYMBOLS = [s for r in ROWS for s in r]


def map_cfg(**extra):
    cfg = {"symbols": SYMBOLS, "layout": [2, 3], "epoch_start": ["epoch"], "epoch_end": ["end"],
           "flash": [{"match": "^R(\\d)$", "row_from_group": 1, "offset": 1},
                     {"match": "^C(\\d)$", "col_from_group": 1, "offset": 1},
                     {"match": "^S(\\d+)$", "symbol_from_group": 1},
                     {"match": "^E(\\d+)/(.+)$", "epoch_group": 1, "symbols_from_group": 2, "sep": ","}],
           "cue": {"match": "^target:(.+)$"}}
    cfg.update(extra)
    return cfg


def test_marker_map_translates_rows_columns_symbols_epochs():
    from eeg_source import MarkerMap
    m = MarkerMap(map_cfg())
    assert m.classify("R2") == ("flash", ["г", "д", "е"], None)
    assert m.classify("C1") == ("flash", ["а", "г"], None)
    assert m.classify("S4") == ("flash", ["д"], None)
    assert m.classify("E7/а,е") == ("flash", ["а", "е"], 7)
    assert m.classify("epoch")[0] == "epoch_start"
    assert m.classify("end")[0] == "epoch_end"
    assert m.classify("target:д") == ("cue", "д", None)
    assert m.classify("something else") == (None, None, None)
    with pytest.raises(ValueError):
        MarkerMap(map_cfg(layout=[3, 3]))


def test_speller_edf_map_matches_speller_annotations():
    from eeg_source import MarkerMap
    m = MarkerMap.load(ROOT / "stimulus_kit" / "tools" / "markers_speller_edf.json")
    assert m.classify("speller/4/17/row/1/2") == ("flash", list("жзийкл"), 4)
    assert m.classify("speller/0/3/column/5/0")[1] == ["е", "л", "с", "ч", "э", "⌫"]
    assert m.classify("speller/2/0/symbol/32/0")[1] == [" "]


def write_edf(path, events, trigger=False):
    """4 EEG channels, a photodiode pulse at every flash, optional trigger channel."""
    import pyedflib
    dur = events[-1][0] + 2
    n = int(dur * FS)
    rng = np.random.default_rng(0)
    eeg = rng.normal(0, 5, (4, n))
    photo, trig = np.zeros(n), np.zeros(n)
    for onset, label in events:
        i = round(onset * FS)
        if label[0] in "RCS":
            photo[i:i + 2] = 100
            trig[i:i + 3] = 10 + int(label[1:])
    names = ["Cz", "Pz", "P3", "P4", "PHOTO"] + (["TRIG"] if trigger else [])
    sigs = list(eeg) + [photo] + ([trig] if trigger else [])
    head = lambda name, dim, lo, hi: {"label": name, "dimension": dim, "sample_frequency": FS,
                                      "physical_min": lo, "physical_max": hi,
                                      "digital_min": -32768, "digital_max": 32767,
                                      "transducer": "", "prefilter": ""}
    with pyedflib.EdfWriter(str(path), len(names), file_type=pyedflib.FILETYPE_EDFPLUS) as w:
        w.set_number_of_annotation_signals(8)      # several markers per 1 s record
        w.setStartdatetime(datetime(2026, 1, 1))
        w.setSignalHeaders([head(c, "uV", -200, 200) for c in names[:4]] + [head("PHOTO", "", 0, 200)]
                           + ([head("TRIG", "", 0, 100)] if trigger else []))
        w.writeSamples(sigs)
        for onset, label in events:
            w.writeAnnotation(onset, 0, label)


def events_two_epochs():
    ev, t = [], 0.5
    for target in ("б", "е"):
        ev += [(t, f"target:{target}"), (t + 0.2, "epoch")]
        t += 0.5
        for label in ("R1", "C2", "R2", "C1", "C3", "S0"):
            ev.append((t, label))
            t += 0.25
        ev.append((t + 0.3, "end"))
        t += 1.0
    return ev


def test_load_reads_units_annotations_and_trigger(tmp_path):
    from eeg_source import load
    path = tmp_path / "rec.edf"
    events = events_two_epochs()
    write_edf(path, events, trigger=True)
    rec = load(str(path))
    assert rec.srate == FS and rec.ch_names[:5] == ["Cz", "Pz", "P3", "P4", "PHOTO"]
    assert rec.units[:4] == ["microvolts"] * 4 and rec.units[4] != "microvolts"
    assert [m.label for m in rec.markers] == [e[1] for e in events]
    assert abs(rec.read(0, FS)[:, 0].std() - 5) < 1
    rec.close()
    trig = load(str(path), trigger_channel="TRIG")
    assert "TRIG" not in trig.ch_names
    codes = [m.code for m in trig.markers if m.code is not None]
    assert codes == [10 + int(e[1][1:]) for e in events if e[1][0] in "RCS"]
    assert any(m.label == "target:б" for m in trig.markers)       # text annotations kept
    trig.close()


def test_replay_over_lsl_keeps_flash_timing_and_contract(tmp_path):
    pylsl = pytest.importorskip("pylsl")
    from eeg_source import EEG_STREAM, MarkerMap, load, replay
    from speller_link import STIMULI_STREAM, CONTROL_STREAM
    path = tmp_path / "rec.edf"
    write_edf(path, events_two_epochs())
    rec = load(str(path))
    tag = uuid.uuid4().hex[:6]
    threading.Thread(target=replay, daemon=True,
                     kwargs=dict(rec=rec, speed=2.0, mmap=MarkerMap(map_cfg()), as_stimulus=True,
                                 as_control=True, wait_consumers=10, log=lambda s: None)).start()

    def inlet(name):
        found = pylsl.resolve_byprop("name", name, 1, 10)
        assert found, name
        i = pylsl.StreamInlet(found[0], processing_flags=pylsl.proc_clocksync)
        i.open_stream(timeout=5)
        return i
    eeg, stim, ctl = inlet(EEG_STREAM), inlet(STIMULI_STREAM), inlet(CONTROL_STREAM)
    x, t, stims, ctls = [], [], [], []
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not any(m.get("cmd") == "end" for m in ctls):
        d, ts = eeg.pull_chunk(timeout=0.0)
        x += d
        t += ts
        for src, out in ((stim, stims), (ctl, ctls)):
            s, ts1 = src.pull_sample(timeout=0.0)
            while s is not None:
                out.append((json.loads(s[0]), ts1) if src is stim else json.loads(s[0]))
                s, ts1 = src.pull_sample(timeout=0.0)
        time.sleep(0.005)
    x, t = np.asarray(x), np.asarray(t)
    assert stims[0][0]["event"] == "keyboard" and stims[0][0]["symbols"] == SYMBOLS, tag
    flashes = [(m, ts) for m, ts in stims if "flash" in m]
    assert len(flashes) == 12
    assert [m["flash"] for m, _ in flashes[:3]] == [ROWS[0], ["б", "д"], ROWS[1]]
    assert [m["epoch"] for m, _ in flashes] == [0] * 6 + [1] * 6
    assert [m["epoch"] for m, _ in stims if m.get("event") == "done"] == [0, 1]
    starts = [m for m in ctls if m.get("cmd") == "epoch_start"]
    assert [m["epoch"] for m in starts] == [0, 1] and all("cue" not in m for m in starts)
    for _, ts in flashes:                     # the photodiode pulse sits exactly at the marker
        k = int(np.argmin(np.abs(t - ts)))
        assert abs(t[k] - ts) < 0.5 / FS
        assert x[k, 4] > 50 and x[k - 1, 4] < 50
