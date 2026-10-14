"""Removable stimulus/decoder modules: protocol speller-link@1."""
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
import pytest

from eeg_speller.core.config import load_config
from eeg_speller.external.controller import ExternalSession, likelihood_vector
from eeg_speller.external.transport import MemoryTransport
from eeg_speller.recording.storage import read_events

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = list(load_config(ROOT / "configs" / "ext_default.yaml").repertoire["symbols"])


def make_session(tmp_path, profile="ext_default.yaml", target="", **external):
    cfg = load_config(ROOT / "configs" / profile)
    if external:
        cfg = cfg.model_copy(update={"external": {**cfg.external, **external}})
    transport = MemoryTransport()
    return ExternalSession(cfg, target, tmp_path, transport=transport), transport


def onehot(symbol, strength=0.6):
    return {s: (strength if s == symbol else (1 - strength) / 35) for s in SYMBOLS}


# ---------------------------------------------------------------- likelihood parsing
def test_likelihood_dict_list_and_missing_symbols():
    p = np.exp(likelihood_vector({"а": 0.5}, SYMBOLS))
    assert p[0] == pytest.approx(0.5, abs=1e-6)
    assert p[1] == pytest.approx(0.5 / 35, abs=1e-6)          # rest shared equally
    q = np.exp(likelihood_vector([1.0] * 36, SYMBOLS))
    assert np.allclose(q, 1 / 36)
    with pytest.raises(ValueError):
        likelihood_vector({"z": 1.0}, SYMBOLS)
    with pytest.raises(ValueError):
        likelihood_vector([1.0] * 5, SYMBOLS)
    with pytest.raises(ValueError):
        likelihood_vector({"а": -1.0}, SYMBOLS)


# ---------------------------------------------------------------- core behaviour
def test_core_types_text_and_never_sends_the_task(tmp_path):
    target = "мир"
    session, tr = make_session(tmp_path, target=target)

    def module(msg):
        if msg["cmd"] == "epoch_start":
            want = target[len(msg["text"])]
            tr.deliver("stimuli", {"epoch": msg["epoch"], "flash": msg["symbols"][:6]})
            tr.deliver("likelihood", {"epoch": msg["epoch"], "p": onehot(want), "final": True})

    tr.listeners.append(module)
    session.start().wait_finished(30)
    state = session.monitor()
    assert state["text"] == target and state["end_reason"] == "target_finished"
    starts = [m for m in tr.sent if m["cmd"] == "epoch_start"]
    assert starts and all(target not in json.dumps(m, ensure_ascii=False) for m in starts)
    assert all("target" not in m for m in tr.sent)
    assert [m["cmd"] for m in tr.sent].count("result") == 3 and tr.sent[-1]["cmd"] == "end"
    integrity = json.loads((Path(state["directory"]) / "derived" / "integrity.json").read_text())
    assert integrity["complete"], integrity


def test_stop_is_sent_once_confident_and_decision_waits_for_final(tmp_path):
    session, tr = make_session(tmp_path, profile="ext_no_llm.yaml", target="а",
                               stop_confidence=0.9, stop_grace_s=5.0, min_updates_before_stop=1)
    stops = []

    def module(msg):
        if msg["cmd"] == "epoch_start":
            e = msg["epoch"]
            tr.deliver("likelihood", {"epoch": e, "p": onehot("а", 0.3)})   # not yet confident
            tr.deliver("likelihood", {"epoch": e, "p": onehot("а", 0.97)})  # confident -> stop
        elif msg["cmd"] == "stop":
            stops.append(msg)
            tr.deliver("likelihood", {"epoch": msg["epoch"], "p": onehot("а", 0.98), "final": True})

    tr.listeners.append(module)
    session.start().wait_finished(20)
    assert len(stops) == 1 and stops[0]["confidence"] >= 0.9
    assert session.monitor()["text"] == "а"
    events = read_events(session.storage.directory)
    kinds = [e["type"] for e in events]
    assert kinds.index("epoch_stop_requested") < kinds.index("symbol_decided")
    post = next(e for e in events if e["type"] == "posterior_computed")
    assert post["payload"]["decided_on"] == "final_likelihood" and post["payload"]["updates"] == 3


def test_stop_disabled_profile_never_sends_stop(tmp_path):
    session, tr = make_session(tmp_path, profile="ext_no_stop.yaml", target="б")

    def module(msg):
        if msg["cmd"] == "epoch_start":
            tr.deliver("likelihood", {"epoch": msg["epoch"], "p": onehot("б", 0.999)})
            tr.deliver("stimuli", {"epoch": msg["epoch"], "event": "done"})
            tr.deliver("likelihood", {"epoch": msg["epoch"], "p": onehot("б", 0.999), "final": True})

    tr.listeners.append(module)
    session.start().wait_finished(20)
    assert "stop" not in [m["cmd"] for m in tr.sent]
    assert session.monitor()["text"] == "б"


def test_unusable_evidence_retries_and_early_answers_are_buffered(tmp_path):
    session, tr = make_session(tmp_path, profile="ext_no_llm.yaml", target="в")
    calls = {"n": 0}

    def module(msg):
        if msg["cmd"] == "epoch_start":
            calls["n"] += 1
            e = msg["epoch"]
            if calls["n"] == 1:
                # an answer for the *next* epoch arrives early and must wait in a buffer
                tr.deliver("likelihood", {"epoch": e + 1, "p": onehot("в", 0.99), "final": True})
                tr.deliver("likelihood", {"epoch": e, "p": onehot("в", 0.99), "final": True, "usable": False})

    tr.listeners.append(module)
    session.start().wait_finished(20)
    state = session.monitor()
    assert state["text"] == "в"
    assert [r["decided"] for r in state["log"]] == [None, "в"]


def test_no_decoder_aborts_epoch_after_done(tmp_path):
    session, tr = make_session(tmp_path, target="г", decode_wait_s=0.05, epoch_timeout_s=5)

    def module(msg):
        if msg["cmd"] == "epoch_start" and msg["epoch"] == 0:
            tr.deliver("stimuli", {"epoch": 0, "event": "done"})
        elif msg["cmd"] == "epoch_start":
            tr.deliver("likelihood", {"epoch": msg["epoch"], "p": onehot("г", 0.99), "final": True})

    tr.listeners.append(module)
    session.start().wait_finished(20)
    kinds = [e["type"] for e in read_events(session.storage.directory)]
    assert "epoch_aborted" in kinds and session.monitor()["text"] == "г"


def test_bad_likelihood_is_rejected_and_logged(tmp_path):
    session, tr = make_session(tmp_path, target="д")

    def module(msg):
        if msg["cmd"] == "epoch_start":
            tr.deliver("likelihood", {"epoch": msg["epoch"], "p": {"q": 1}})
            tr.deliver("likelihood", {"epoch": msg["epoch"], "p": onehot("д", 0.99), "final": True})

    tr.listeners.append(module)
    session.start().wait_finished(20)
    events = read_events(session.storage.directory)
    assert any(e["type"] == "warning" and "rejected likelihood" in e["payload"]["message"] for e in events)
    assert session.monitor()["text"] == "д"


# ---------------------------------------------------------------- real LSL, separate module
def test_lsl_end_to_end_and_marker_timing(tmp_path, monkeypatch):
    pylsl = pytest.importorskip("pylsl")
    monkeypatch.setenv("LSLAPICFG", str(ROOT / "tests" / "lsl_local.cfg"))
    sys.path.insert(0, str(ROOT / "stimulus_kit"))
    sys.path.insert(0, str(ROOT / "stimulus_kit" / "examples"))
    import speller_link
    import headless_sim

    cfg = load_config(ROOT / "configs" / "ext_default.yaml")
    cfg = cfg.model_copy(update={"stimulus": {**cfg.stimulus, "pause_ms": 500}})
    session = ExternalSession(cfg, "да", tmp_path).start()
    link = speller_link.SpellerLink(name="pytest_sim", role="stimulus+decoder", wait_for_speller=10)
    sent_times = []
    original = link.flash

    def recording_flash(symbols, timestamp=None, epoch=None):
        sent_times.append(original(symbols, timestamp, epoch))
        return sent_times[-1]

    link.flash = recording_flash
    worker = threading.Thread(target=headless_sim.run, args=("да",),
                              kwargs={"d_prime": 2.0, "flash_ms": 50, "isi_ms": 30, "link": link,
                                      "verbose": False}, daemon=True)
    worker.start()
    assert session.wait_finished(90)
    state = session.monitor()
    assert state["text"] == "да", state
    received = [e["payload"]["onset_lsl_s"] for e in read_events(session.storage.directory)
                if e["type"] == "stimulus_event"]
    assert len(received) == len(sent_times) > 0
    # Onsets travel as LSL timestamps of the flip itself: transport delay does not
    # shift them. On one computer the clock offset is zero, so the error is ~0.
    error_ms = max(abs(a - b) for a, b in zip(sorted(received), sorted(sent_times))) * 1000
    assert error_ms < 1.0, error_ms


def test_wait_epoch_does_not_skip_an_epoch_that_started_meanwhile():
    pytest.importorskip("pylsl")
    sys.path.insert(0, str(ROOT / "stimulus_kit"))
    import speller_link
    link = speller_link.SpellerLink(name="pytest_skip", wait_for_speller=0)
    link._handle({"cmd": "epoch_start", "epoch": 0, "symbols": SYMBOLS})
    assert link.wait_epoch(timeout=0.1).id == 0
    link._handle({"cmd": "epoch_start", "epoch": 1, "symbols": SYMBOLS})   # arrives while busy
    assert link.wait_epoch(timeout=0.1).id == 1
    assert link.wait_epoch(timeout=0.1) is None                            # nothing new
    link._handle({"cmd": "stop", "epoch": 1})
    assert link.should_stop()
    link._handle({"cmd": "end"})
    assert link.ended and link.wait_epoch(timeout=0.1) is None
