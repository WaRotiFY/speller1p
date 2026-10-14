"""Speller side of the ``speller-link@1`` protocol (see stimulus_kit/speller_link.py).

The core publishes commands and receives two kinds of messages:
- ``stimuli``: flash markers / "done" events from stimulus modules (logged only);
- ``likelihood``: symbol likelihoods from decoder modules (fused with the LLM prior).

The core never draws or schedules stimuli. Any number of modules may come and go;
new streams are discovered continuously.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import queue
import threading
import time

PROTOCOL = "speller-link@1"
CONTROL_STREAM = "SpellerControl"
STIMULI_STREAM = "SpellerStimuli"
LIKELIHOOD_STREAM = "SpellerLikelihood"


@dataclass(frozen=True)
class Incoming:
    kind: str            # "stimuli" | "likelihood"
    msg: dict
    lsl_time: float      # sender timestamp mapped to this computer's LSL clock
    source: str          # source_id of the sending stream
    module: str          # module name declared by the sender


class MemoryTransport:
    """In-process transport for tests: modules call ``deliver``; commands go to ``sent``."""

    def __init__(self):
        self.sent: list[dict] = []
        self.inbox: "queue.Queue[Incoming]" = queue.Queue()
        self.listeners = []

    def send(self, msg: dict) -> None:
        self.sent.append(msg)
        for listener in list(self.listeners):
            listener(msg)

    def deliver(self, kind: str, msg: dict, lsl_time: float | None = None, module: str = "test") -> None:
        self.inbox.put(Incoming(kind, msg, time.monotonic() if lsl_time is None else lsl_time, module, module))

    def poll(self, timeout: float) -> list[Incoming]:
        out = []
        try:
            out.append(self.inbox.get(timeout=timeout))
            while True:
                out.append(self.inbox.get_nowait())
        except queue.Empty:
            pass
        return out

    def consumers(self) -> bool:
        return True

    def status(self) -> dict:
        return {"transport": "memory", "modules": []}

    def close(self) -> None:
        pass


class LSLTransport:
    """Real transport over Lab Streaming Layer (marker streams of JSON strings)."""

    def __init__(self, session_id: str):
        try:
            import pylsl
        except (ImportError, RuntimeError) as exc:
            raise ValueError(f"LSL unavailable: {exc}; install pylsl") from exc
        self.lsl = pylsl
        info = pylsl.StreamInfo(CONTROL_STREAM, "Markers", 1, pylsl.IRREGULAR_RATE,
                                pylsl.cf_string, "speller-control-" + session_id)
        info.desc().append_child_value("protocol", PROTOCOL)
        self.outlet = pylsl.StreamOutlet(info)
        self.inbox: "queue.Queue[Incoming]" = queue.Queue()
        self.inlets: dict[str, dict] = {}
        self.closed = threading.Event()
        self.lock = threading.Lock()
        self.resolvers = {kind: pylsl.ContinuousResolver("name", name, forget_after=5.0)
                          for kind, name in (("stimuli", STIMULI_STREAM), ("likelihood", LIKELIHOOD_STREAM))}
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------- outgoing
    def send(self, msg: dict) -> None:
        self.outlet.push_sample([json.dumps(msg, ensure_ascii=False)], self.lsl.local_clock())

    def consumers(self) -> bool:
        return self.outlet is not None and self.outlet.have_consumers()

    # ------------------------------------------------------------- incoming
    def _attach(self):
        for kind, resolver in self.resolvers.items():
            for info in resolver.results():
                sid = info.source_id()
                if sid in self.inlets:
                    continue
                # Clock synchronization maps the sender's LSL timestamps into this
                # computer's LSL clock (sub-millisecond on a local network).
                inlet = self.lsl.StreamInlet(info, max_buflen=60, recover=False,
                                             processing_flags=self.lsl.proc_clocksync)
                inlet.open_stream(timeout=2)
                full = inlet.info(timeout=2)
                module = full.desc().child_value("module") or sid
                with self.lock:
                    self.inlets[sid] = {"kind": kind, "inlet": inlet, "module": module,
                                        "since": time.monotonic(), "messages": 0, "last": None}

    def _run(self):
        last_scan = 0.0
        while not self.closed.is_set():
            if time.monotonic() - last_scan > 0.5:
                try:
                    self._attach()
                except Exception:
                    pass
                last_scan = time.monotonic()
            idle = True
            with self.lock:
                items = list(self.inlets.items())
            for sid, entry in items:
                try:
                    samples, stamps = entry["inlet"].pull_chunk(timeout=0.0)
                except Exception:           # the module closed its stream
                    with self.lock:
                        self.inlets.pop(sid, None)
                    continue
                for sample, stamp in zip(samples, stamps):
                    idle = False
                    try:
                        msg = json.loads(sample[0])
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(msg, dict):
                        continue
                    entry["messages"] += 1
                    entry["last"] = time.monotonic()
                    self.inbox.put(Incoming(entry["kind"], msg, float(stamp), sid, entry["module"]))
            if idle:
                time.sleep(0.002)

    def poll(self, timeout: float) -> list[Incoming]:
        out = []
        try:
            out.append(self.inbox.get(timeout=timeout))
            while True:
                out.append(self.inbox.get_nowait())
        except queue.Empty:
            pass
        return out

    def status(self) -> dict:
        now = time.monotonic()
        with self.lock:
            modules = [{"module": e["module"], "kind": e["kind"], "source_id": sid,
                        "messages": e["messages"],
                        "last_message_s": None if e["last"] is None else round(now - e["last"], 2)}
                       for sid, e in self.inlets.items()]
        return {"transport": "lsl", "control_listeners": self.consumers(), "modules": modules}

    def close(self) -> None:
        self.closed.set()
        self.thread.join(timeout=2)
        with self.lock:
            for entry in self.inlets.values():
                try:
                    entry["inlet"].close_stream()
                except Exception:
                    pass
            self.inlets.clear()
        self.outlet = None
