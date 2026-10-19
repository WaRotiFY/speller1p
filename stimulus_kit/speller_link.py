"""speller_link — the only file a stimulus or decoder module needs from the speller.

Copy this file next to your program. It depends only on ``pylsl``
(``pip install pylsl``). Protocol name: ``speller-link@1``.

Minimum rule for a stimulus program (everything else is optional)::

    from speller_link import SpellerLink
    link = SpellerLink()
    ...
    pygame.display.flip()
    link.flash(["а", "б", "в", "г", "д", "е"])   # right after the flip

Optional calls::

    epoch = link.wait_epoch()     # block until the speller asks for the next symbol
    link.should_stop()            # True when the speller is confident enough -> next symbol
    link.done()                   # you finished presenting this epoch
    link.result()                 # last chosen symbol and typed text, to show the participant
    link.send_likelihood({...})   # only if your program also decodes EEG

All messages are JSON strings on three LSL marker streams:

    SpellerControl     speller -> modules    epoch_start / stop / result / pause / resume / end
    SpellerStimuli     stimulus -> speller   {"epoch": N, "flash": [symbols]}   (+ {"epoch": N, "event": "done"})
    SpellerLikelihood  decoder  -> speller   {"epoch": N, "p": {symbol: value}, "final": bool}

The LSL timestamp of a ``flash`` sample is the moment of the flip in the LSL clock
of the stimulus computer. The speller never looks at the screen.
"""
import json
import threading
import time

try:
    import pylsl
except ImportError as exc:  # pragma: no cover - message for module authors
    raise ImportError("speller_link needs pylsl: pip install pylsl") from exc

PROTOCOL = "speller-link@1"
CONTROL_STREAM = "SpellerControl"
STIMULI_STREAM = "SpellerStimuli"
LIKELIHOOD_STREAM = "SpellerLikelihood"


class Epoch:
    """What the speller asks for: choose one symbol out of ``symbols``."""

    def __init__(self, msg):
        self.id = int(msg["epoch"])
        self.symbols = list(msg["symbols"])
        self.layout = tuple(msg.get("layout") or (6, 6))
        self.text = msg.get("text", "")
        self.settings = dict(msg.get("settings") or {})
        # Symbol to attend, sent only in copy-spelling / calibration sessions; else None.
        self.cue = msg.get("cue")

    def rows(self):
        """Symbols split into rows of the layout (a convenience for matrix displays)."""
        cols = self.layout[1]
        return [self.symbols[i:i + cols] for i in range(0, len(self.symbols), cols)]

    def cell(self, symbol):
        """(row, column) of ``symbol`` in the layout."""
        i = self.symbols.index(symbol)
        return divmod(i, self.layout[1])

    def __repr__(self):
        return "Epoch(id=%d, text=%r)" % (self.id, self.text)


class SpellerLink:
    """Connection of one stimulus and/or decoder module to the speller.

    ``role`` only names the module in the speller's log ("stimulus", "decoder",
    or both: "stimulus+decoder"). ``wait_for_speller`` blocks until the speller
    publishes its control stream; set it to 0 to start without a speller.
    """

    def __init__(self, name="stimulus", role="stimulus", wait_for_speller=60.0):
        self.name = name
        self.role = role
        uid = "%s-%d" % (name, int(time.time() * 1000))
        stim = pylsl.StreamInfo(STIMULI_STREAM, "Markers", 1, pylsl.IRREGULAR_RATE,
                                pylsl.cf_string, "stim-" + uid)
        stim.desc().append_child_value("protocol", PROTOCOL)
        stim.desc().append_child_value("module", name)
        self._stim = pylsl.StreamOutlet(stim)
        self._lik = None
        self._lock = threading.Lock()
        self._epoch = None              # current Epoch
        self._returned = None           # id of the last epoch handed out by wait_epoch
        self._stopped = set()           # epochs the speller asked to stop
        self._result = None             # {"epoch", "symbol", "text", ...}
        self._paused = False
        self._ended = False
        self._new_epoch = threading.Condition(self._lock)
        self._inlet = None
        self._closing = threading.Event()
        if "decoder" in role:
            self._open_likelihood()
        if wait_for_speller:
            self._connect(wait_for_speller)

    # ------------------------------------------------------------ connection
    def _connect(self, timeout):
        found = pylsl.resolve_byprop("name", CONTROL_STREAM, 1, timeout)
        if not found:
            raise RuntimeError("speller control stream %r not found; start the speller first"
                               % CONTROL_STREAM)
        self._inlet = pylsl.StreamInlet(found[0], max_buflen=60, recover=True)
        self._inlet.open_stream(timeout=5)
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        while not self._closing.is_set():
            try:
                sample, _ = self._inlet.pull_sample(timeout=0.1)
            except Exception:           # lost stream: behave as if the session ended
                sample = None
                with self._lock:
                    self._ended = True
                    self._new_epoch.notify_all()
                return
            if sample is None:
                continue
            try:
                msg = json.loads(sample[0])
            except ValueError:
                continue
            self._handle(msg)

    def _handle(self, msg):
        cmd = msg.get("cmd")
        with self._lock:
            if cmd == "epoch_start":
                self._epoch = Epoch(msg)
                self._new_epoch.notify_all()
            elif cmd == "stop":
                self._stopped.add(int(msg.get("epoch", -1)))
            elif cmd == "result":
                self._result = msg
            elif cmd == "pause":
                self._paused = True
            elif cmd == "resume":
                self._paused = False
            elif cmd == "end":
                self._ended = True
                self._new_epoch.notify_all()

    # -------------------------------------------------------- stimulus calls
    def flash(self, symbols, timestamp=None, epoch=None):
        """Report that ``symbols`` just became visible.

        Call immediately after ``display.flip()``. ``timestamp`` may be given in
        ``pylsl.local_clock()`` seconds if you measured the flip yourself (e.g.
        with a photodiode); otherwise "now" is used.
        """
        if timestamp is None:
            timestamp = pylsl.local_clock()
        msg = {"epoch": self._epoch_id(epoch), "flash": list(symbols)}
        self._stim.push_sample([json.dumps(msg, ensure_ascii=False)], timestamp)
        return timestamp

    def done(self, epoch=None):
        """Tell the speller that presentation of this epoch is finished."""
        msg = {"epoch": self._epoch_id(epoch), "event": "done"}
        self._stim.push_sample([json.dumps(msg)], pylsl.local_clock())

    def wait_epoch(self, timeout=None):
        """Block until the speller starts a new epoch; ``None`` when the session ends."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._lock:
            # "New" means not yet returned by wait_epoch, so an epoch that started
            # while you were busy (e.g. showing the result) is never skipped.
            while not self._ended and (self._epoch is None or self._epoch.id == self._returned):
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    return None
                self._new_epoch.wait(0.1 if left is None else min(left, 0.1))
            if self._ended:
                return None
            self._returned = self._epoch.id
            return self._epoch

    def should_stop(self, epoch=None):
        """True when the speller is confident (or the session ended): move on."""
        with self._lock:
            return self._ended or self._epoch_id(epoch) in self._stopped

    @property
    def epoch(self):
        return self._epoch

    @property
    def paused(self):
        return self._paused

    @property
    def ended(self):
        return self._ended

    def result(self):
        """Last decision: dict with ``symbol``, ``text``, ``confidence`` (or None)."""
        with self._lock:
            return dict(self._result) if self._result else None

    def wait_result(self, epoch=None, timeout=10.0):
        """Wait for the speller's decision about ``epoch`` (default: current)."""
        eid = self._epoch_id(epoch)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._ended:
            res = self.result()
            if res and res.get("epoch") == eid:
                return res
            time.sleep(0.01)
        return None

    # --------------------------------------------------------- decoder calls
    def send_likelihood(self, values, final=False, epoch=None, usable=True, info=None):
        """Send P(EEG | target = symbol) for each symbol — WITHOUT any language model.

        ``values``: dict {symbol: probability} or a list in the order of
        ``epoch.symbols``. Values need not sum to 1. Symbols missing from a dict
        share the remaining mass equally. Send updates as evidence accumulates;
        mark the last one ``final=True``.
        """
        if self._lik is None:
            self._open_likelihood()
            time.sleep(1.0)             # first use: let the speller discover the new stream
        if not isinstance(values, dict):
            values = [float(v) for v in values]
        else:
            values = {str(k): float(v) for k, v in values.items()}
        msg = {"epoch": self._epoch_id(epoch), "p": values, "final": bool(final),
               "usable": bool(usable)}
        if info:
            msg["info"] = info
        self._lik.push_sample([json.dumps(msg, ensure_ascii=False)], pylsl.local_clock())

    def _open_likelihood(self):
        info = pylsl.StreamInfo(LIKELIHOOD_STREAM, "Markers", 1, pylsl.IRREGULAR_RATE,
                                pylsl.cf_string, "lik-%s-%d" % (self.name, int(time.time() * 1000)))
        info.desc().append_child_value("protocol", PROTOCOL)
        info.desc().append_child_value("module", self.name)
        self._lik = pylsl.StreamOutlet(info)

    # ----------------------------------------------------------------- misc
    def _epoch_id(self, epoch):
        if epoch is not None:
            return int(epoch.id if isinstance(epoch, Epoch) else epoch)
        return self._epoch.id if self._epoch is not None else -1

    def close(self):
        self._closing.set()


def now():
    """LSL clock, the time base of every flash timestamp."""
    return pylsl.local_clock()
