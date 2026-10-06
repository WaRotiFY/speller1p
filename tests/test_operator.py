"""Experimenter window: settings, live state, synchronisation and HTTP API."""
import json
from pathlib import Path
import threading
import time
from http.server import ThreadingHTTPServer
import urllib.error
import urllib.request

import numpy as np
from pytest import raises

from eeg_speller.recording.storage import check_integrity, read_events
from eeg_speller.web import settings
from eeg_speller.web.engine import InteractiveSession, SessionManager
from eeg_speller.web.server import make_handler


def press_for(session, symbol):
    target = session.rep.index(symbol)
    return [target in flash["indices"] for flash in session.plan]


def test_settings_whitelist_and_validation():
    cfg = settings.build_config("gui_slow.yaml", {"fusion.alpha": 0.5, "stimulus.flash_ms": 200,
                                                  "detector.mode": "label", "slow_llm.trigger": "manual"},
                                participant_id="P001", task_id="T001")
    assert cfg.fusion["alpha"] == 0.5 and cfg.stimulus["flash_ms"] == 200
    assert cfg.detector["mode"] == "label" and cfg.run_mode == "live" and cfg.llm["allow_remote"] is False
    bad = [{"seed": 1}, {"llm.allow_remote": True}, {"stimulus.flash_ms": 205}, {"stimulus.flash_ms": 40},
           {"fusion.strategy": "max"}, {"epoch.n_repetitions": 2.5}, {"slow_llm.strict_word_count": "yes"},
           {"gui.response_false_alarm": 0.9}, {"slow_llm.backend": "openai_compat"}]
    for override in bad:
        with raises(ValueError):
            settings.build_config("gui_slow.yaml", override, participant_id="P001", task_id="T001")
    with raises(ValueError):
        settings.build_config("../default.yaml", {}, participant_id="P001", task_id="T001")
    with raises(ValueError):
        settings.check_identifier("../x", "participant_id")
    values = settings.profile_values("gui_slow.yaml")
    assert set(values) == {field.key for field in settings.FIELDS}
    assert values["slow_llm.every_k_symbols"] == 10


def test_no_llm_session_has_uniform_prior_and_no_predictions(tmp_path: Path):
    session = InteractiveSession("gui_slow.yaml", "а", tmp_path, overrides={"mode.llm": "off"},
                                 participant_id="P002", task_id="CUSTOM")
    current = session.monitor()["current"]
    assert np.allclose(current["prior"], 1 / 36, atol=1e-4)
    assert current["words"] == [] and session.slow is None
    session.submit(press_for(session, "а"))
    events = read_events(session.storage.directory)
    assert not any(e["type"] in ("word_candidates", "correction_requested") for e in events)
    assert session.storage.manifest["llm_mode"] == "off"


def test_live_distribution_follows_flash_progress(tmp_path: Path):
    session = InteractiveSession("gui_slow.yaml", "п", tmp_path, participant_id="P003", task_id="CUSTOM")
    eid = session.epoch_id
    before = session.monitor()["current"]
    assert before["likelihood"] is None and before["evidence_flashes"] == 0
    session.progress_event(eid, "epoch_begin")
    hits = press_for(session, "п")
    for i, hit in enumerate(hits):
        session.progress_event(eid, "flash", i)
        if hit:
            session.progress_event(eid, "hit", i)
    session.progress_event(eid, "flashes_done")
    current = session.monitor()["current"]
    assert current["evidence_flashes"] == len(hits)
    assert current["leader"]["symbol"] == "п" and current["leader"]["confidence"] > 0.9
    assert current["hit_flashes"] == [i for i, h in enumerate(hits) if h]
    assert not session.progress_event(eid + 5, "flash", 0)          # stale epoch is ignored
    with raises(ValueError):
        session.progress_event(eid, "flash", 999)
    state = session.submit(hits, epoch_id=eid)
    assert state["finished"] and state["text"] == "п"
    row = session.monitor()["log"][-1]
    assert row["correct"] is True and row["rank_target"] == 1 and row["target"] == "п"


def test_context_source_and_detector_label_mode(tmp_path: Path):
    session = InteractiveSession("gui_slow.yaml", "", tmp_path, participant_id="P004", task_id="FREE",
                                 overrides={"fast_llm.context_source": "secondary", "detector.mode": "label",
                                            "detector.tau": 0.99})
    weak = [False] * len(session.plan)
    weak[0] = True                                         # one press: low confidence
    state = session.submit(weak)
    assert state["text"] != ""                             # label mode decides anyway
    assert session.log[-1]["uncertain"] is True
    events = read_events(session.storage.directory)
    prior = [e for e in events if e["type"] == "prior_computed"][0]
    assert prior["payload"]["context_source"] == "secondary"
    session.stop()


def test_manager_pause_stop_and_epoch_guard(tmp_path: Path):
    manager = SessionManager(tmp_path)
    state = manager.start("gui_slow.yaml", "аб", participant_id="P005", task_id="CUSTOM",
                          experiment_id="EXP-T")
    sid = state["session_id"]
    with raises(ValueError):
        manager.start("gui_slow.yaml", "", participant_id="P006", task_id="FREE")
    session = manager.get(sid)
    version = manager.version
    manager.pause(sid, True)
    assert manager.version > version and manager.live("participant")["session"]["paused"]
    manager.pause(sid, False)
    hits = press_for(session, "а")
    manager.submit(sid, hits, 0)
    with raises(ValueError):
        manager.submit(sid, hits, 0)                       # second window, same epoch
    manager.stop(sid, "operator_stopped")
    assert manager.live()["session"]["finished"]
    events = read_events(session.storage.directory)
    assert [e["type"] for e in events if "paused" in e["type"] or "resumed" in e["type"]] == \
        ["session_paused", "session_resumed"]
    assert events[-1]["payload"]["reason"] == "operator_stopped"
    assert session.storage.directory.parent.name == "EXP-T"
    assert check_integrity(session.storage.directory)["complete"]


def test_wait_wakes_on_change_and_times_out(tmp_path: Path):
    manager = SessionManager(tmp_path)
    started = time.monotonic()
    manager.wait(manager.version, 0.2)
    assert time.monotonic() - started >= 0.15
    timer = threading.Timer(0.1, manager.bump)
    timer.start()
    started = time.monotonic()
    manager.wait(manager.version, 5)
    assert time.monotonic() - started < 2


def test_http_operator_flow(tmp_path: Path):
    manager = SessionManager(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(manager))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path):
        with urllib.request.urlopen(base + path, timeout=10) as response:
            return response.status, response.read()

    def post(path, body):
        request = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    try:
        assert get("/operator")[0] == 200 and get("/operator.js")[0] == 200
        init = json.loads(get("/api/operator/init")[1])
        assert "gui_slow.yaml" in init["profiles"] and init["tasks"][0]["id"] == "T001"
        with raises(urllib.error.HTTPError):
            post("/api/start", {"profile": "gui_slow.yaml", "participant_id": "P007",
                                "task": {"kind": "task", "id": "T001"}, "settings": {"seed": 3}})
        state = post("/api/start", {"profile": "gui_slow.yaml", "participant_id": "P007",
                                    "task": {"kind": "custom", "text": "Ёж "},
                                    "settings": {"stimulus.flash_ms": 100}})
        assert state["target"] == "еж " and state["task_id"] == "CUSTOM"
        live = json.loads(get(f"/api/live?role=participant&since=-1&timeout=0")[1])
        assert live["session"]["session_id"] == state["session_id"] and "log" not in live["session"]
        assert post("/api/progress", {"session_id": state["session_id"], "epoch_id": 0,
                                      "event": "epoch_begin"})["accepted"]
        operator = json.loads(get(f"/api/live?role=operator&since=-1&timeout=0")[1])
        assert operator["participant_connected"] and operator["session"]["current"]["phase"] == "flashing"
        stopped = post("/api/stop", {"session_id": state["session_id"], "by": "operator"})
        assert stopped["finished"] and stopped["end_reason"] == "operator_stopped"
    finally:
        server.shutdown()
        server.server_close()
