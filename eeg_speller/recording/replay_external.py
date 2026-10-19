"""Replay a recorded operator-window session with other language models or fusion.

The EEG side is fixed: replay epoch k receives exactly the stimulus marks and likelihood
updates that were recorded in epoch k of the source session, on the keyboard the module
announced. Everything else is recomputed by the same controller as in a live session:
fast-LLM prior, fusion plugin, early stop, decision, typed text, slow-LLM corrections.

    python -m eeg_speller replay runs/external/<session>
    python -m eeg_speller replay runs/external/<session> --config my_changes.yaml --llm off

``--config`` is a partial YAML merged over the session's own ``config.yaml``, e.g.::

    fast_llm: {backend: ngram, model_file: ru_char4.npz, model_id: ru_char4}
    fusion: {plugin: log_pool, params: {alpha: 0.5}}
    external: {stop_confidence: 0.95}

Limits (see docs/research/OPEN_QUESTIONS.md, Q-48): the participant attended to what the
original session asked for, so once the replayed text diverges from the original, the
recorded evidence no longer matches what a participant would have looked at. An earlier
stop is honoured (the evidence received so far becomes final); a later stop cannot add
evidence that was never recorded.
"""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import yaml

from eeg_speller.core.config import Config, _merge, load_config
from eeg_speller.external.controller import ExternalSession
from eeg_speller.external.transport import MemoryTransport
from eeg_speller.recording.storage import read_events

FAST = {"pause_ms": 0, "decode_wait_s": 0.3, "stop_grace_s": 0.3, "wait_modules_s": 5.0, "keyboard_wait_s": 1.0}


def is_external(session_dir: Path) -> bool:
    for line in (Path(session_dir) / "events.jsonl").open(encoding="utf-8"):
        event = json.loads(line)
        if event["type"] == "session_started":
            return event["payload"].get("paradigm") == "external"
    return False


def recorded_epochs(session_dir: Path):
    """(keyboard, epochs): per recorded epoch the incoming messages in arrival order."""
    events = read_events(Path(session_dir))
    keyboard = None
    for e in events:
        if e["type"] == "session_started" and keyboard is None:
            keyboard = e["payload"]["keyboard"]
        if e["type"] == "keyboard_set":
            keyboard = {"symbols": e["payload"]["symbols"], "layout": e["payload"]["layout"]}
    order = [e["epoch_id"] for e in events if e["type"] == "epoch_started"]
    items: dict[int, list] = {eid: [] for eid in order}
    for e in events:
        eid, p = e.get("epoch_id"), e["payload"]
        if eid not in items:
            continue
        if e["type"] == "stimulus_event":
            items[eid].append((p.get("recv_lsl_s") or 0.0, "stimuli",
                               {"flash": p["symbols"]}, p.get("onset_lsl_s")))
        elif e["type"] == "physiology_observation":
            mm = p.get("module_metrics") or {}
            items[eid].append((p.get("recv_lsl_s") or 0.0, "likelihood",
                               {"p": p.get("raw") if p.get("raw") is not None else p["log_p"],
                                "final": bool(mm.get("final")), "usable": bool(mm.get("usable", True))},
                               p.get("sent_lsl_s")))
    epochs = [sorted(items[eid], key=lambda x: x[0]) for eid in order]
    return keyboard, epochs


class _Player:
    """Plays the recorded epochs as a stimulus module + decoder would."""

    def __init__(self, transport: MemoryTransport, epochs):
        self.tr, self.epochs, self.next = transport, epochs, 0
        self.session: ExternalSession | None = None
        self.stop = threading.Event()

    def __call__(self, msg):
        if msg.get("cmd") == "epoch_start":
            if self.next >= len(self.epochs):
                self.session.request_stop("replay_finished")
                return
            self.stop.clear()
            threading.Thread(target=self._play, args=(msg["epoch"], self.epochs[self.next]), daemon=True).start()
            self.next += 1
        elif msg.get("cmd") == "stop":
            self.stop.set()

    def _wait_consumed(self, flashes, updates):
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            cur = self.session.current
            if self.tr.inbox.empty() and cur is not None and cur["flashes"] >= flashes and cur["updates"] >= updates:
                return
            time.sleep(0.001)

    def _play(self, epoch, items):
        flashes = updates = 0
        last = None
        for _, kind, msg, stamp in items:
            if self.stop.is_set():
                break
            self.tr.deliver(kind, {"epoch": epoch, **msg}, lsl_time=stamp, module="replay")
            if kind == "stimuli":
                flashes += 1
            else:
                updates += 1
                last = msg
                if msg["final"]:
                    break
            self._wait_consumed(flashes, updates)
        if self.stop.is_set() and last is not None and not last["final"]:
            # The speller stopped earlier than the original: the evidence so far becomes final.
            self.tr.deliver("likelihood", {"epoch": epoch, **last, "final": True}, module="replay")
        self.tr.deliver("stimuli", {"epoch": epoch, "event": "done"}, module="replay")


def replay_external(session_dir, output="runs", config=None, llm=None) -> dict:
    # IMPROV[Q-48]
    src = Path(session_dir)
    data = load_config(src / "config.yaml").model_dump()
    if config:
        data = _merge(data, yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {})
    if llm in ("on", "off"):
        data["mode"]["llm"] = llm
    data["run_mode"] = "replay"
    cfg = Config.model_validate(data)
    cfg = cfg.model_copy(update={"stimulus": {**cfg.stimulus, "pause_ms": FAST["pause_ms"]},
                                 "external": {**(cfg.external or {}),
                                              **{k: v for k, v in FAST.items() if k != "pause_ms"}}})
    target = (src / "texts" / "target.txt").read_text(encoding="utf-8")
    keyboard, epochs = recorded_epochs(src)
    tr = MemoryTransport()
    player = _Player(tr, epochs)
    tr.listeners.append(player)
    session = ExternalSession(cfg, target, Path(output), transport=tr, replay_of=src.name,
                              passport={"replay_of": src.name})
    player.session = session
    tr.deliver("stimuli", {"event": "keyboard", **keyboard}, module="replay")
    session.start()
    session.wait_finished()
    state = session.monitor()
    replayed = [r for r in session.log if r["decided"] is not None]
    return {"session": str(session.storage.directory), "source": str(src), "target": target,
            "original": _summary(src), "error": state["error"],
            "replay": {"text": state["text"], "corrected": state["corrected"], "epochs": len(session.log),
                       "accuracy": state["stats"]["accuracy"], "early_stops": state["stats"]["early_stops"],
                       "mean_flashes": round(sum(r["flashes"] for r in replayed) / len(replayed), 1)
                       if replayed else None}}


def _summary(src: Path) -> dict:
    events = read_events(src)
    targets = {e["epoch_id"]: e["payload"].get("target_symbol") for e in events if e["type"] == "epoch_started"}
    decided = [e for e in events if e["type"] == "symbol_decided" and e["payload"].get("symbol")]
    judged = [e for e in decided if targets.get(e["epoch_id"]) is not None]
    flashes: dict = {}
    for e in events:
        if e["type"] == "stimulus_event":
            flashes[e["epoch_id"]] = flashes.get(e["epoch_id"], 0) + 1
    text = src / "texts" / "primary_final.txt"
    corrected = src / "texts" / "secondary_final.txt"
    return {"text": text.read_text(encoding="utf-8") if text.is_file() else None,
            "corrected": corrected.read_text(encoding="utf-8") if corrected.is_file() else None,
            "epochs": len(targets),
            "accuracy": round(sum(e["payload"]["typed"] == targets[e["epoch_id"]] for e in judged) / len(judged), 3)
            if judged else None,
            "early_stops": sum(e["type"] == "epoch_stop_requested" for e in events),
            "mean_flashes": round(sum(flashes.get(e["epoch_id"], 0) for e in decided) / len(decided), 1)
            if decided else None}
