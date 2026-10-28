"""EEG source for the speller: play any EDF/EDF+/BDF recording, or relay a live amplifier, into LSL.

Physiology is three separate programs connected only by LSL:

    stimulation  (stimulus_kit/modules/…)      -> SpellerStimuli
    EEG source   (this tool)                   -> SpellerEEG
    decoder      (stimulus_kit/decoders/…)     -> SpellerLikelihood

The decoder cannot tell a recording from a live amplifier: it reads the EEG stream
and cuts windows by the LSL time of the flash markers. Run it with
``--eeg-name SpellerEEG``.

Unlike ``edf_to_lsl.py``, which plays only EDFs recorded by this speller, this tool
reads third-party recordings: markers come from EDF+ annotations or a trigger
channel, and a JSON marker map (``--marker-map``) translates the lab's codes into
speller-link messages. See ``docs/EEG_SOURCE.md``.

    python stimulus_kit/tools/eeg_source.py info recording.edf
    python stimulus_kit/tools/eeg_source.py replay recording.edf
    python stimulus_kit/tools/eeg_source.py replay recording.edf --marker-map map.json --as-stimulus
    python stimulus_kit/tools/eeg_source.py replay recording.edf --trigger-channel TRIG --marker-map map.json --as-stimulus --as-control
    python stimulus_kit/tools/eeg_source.py live --find type=EEG
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pylsl  # noqa: E402
from speller_link import CONTROL_STREAM, PROTOCOL, SpellerLink  # noqa: E402

EEG_STREAM = "SpellerEEG"
REPLAY_MARKERS_STREAM = "SpellerReplayMarkers"
CHUNK_S = 0.02                     # EEG goes out in 20 ms chunks
VOLT_TO_UV = {"uv": 1.0, "µv": 1.0, "μv": 1.0, "mv": 1e3, "v": 1e6, "nv": 1e-3}


# ──────────────────────────────── EDF reading ────────────────────────────────

@dataclass
class Marker:
    onset_s: float                 # seconds from the start of the file
    label: str                     # annotation text, or trigger code as text
    code: int | None = None
    duration_s: float = 0.0


@dataclass
class Recording:
    path: str
    srate: float
    ch_names: list[str]
    units: list[str]               # "microvolts" for voltage channels, file unit otherwise
    n_samples: int
    markers: list[Marker]
    start: str | None
    _reader: object = field(repr=False, default=None)
    _idx: list[int] = field(repr=False, default_factory=list)
    _scale: np.ndarray = field(repr=False, default=None)

    def close(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def read(self, start: int, stop: int) -> np.ndarray:
        """Samples [start, stop) as (n, channels); voltage channels in µV."""
        n = stop - start
        out = np.empty((n, len(self._idx)), np.float32)
        for j, i in enumerate(self._idx):
            out[:, j] = self._reader.readSignal(i, start, n) * self._scale[j]
        return out


def _open(path):
    import pyedflib                # EDF, EDF+ and BDF (24-bit) alike
    return pyedflib.EdfReader(path)


def _unit(reader, i):
    return reader.getPhysicalDimension(i).strip()


def _numeric(text):
    return int(text) if re.fullmatch(r"-?\d+", text.strip()) else None


def edges_from_trigger(x: np.ndarray, srate: float) -> list[Marker]:
    """One marker per change of the trigger channel to a non-zero value."""
    x = np.round(x).astype(np.int64)
    prev = np.concatenate([[0], x[:-1]])
    idx = np.nonzero((x != prev) & (x != 0))[0]
    return [Marker(onset_s=i / srate, label=str(int(x[i])), code=int(x[i])) for i in idx]


def trigger_candidates(reader) -> list[str]:
    """Channels that look like triggers: by name, or a few integer values only."""
    names = []
    for i, name in enumerate(reader.getSignalLabels()):
        if re.search(r"trig|stim|marker|status|event|sync|photo", name, re.I):
            names.append(name)
            continue
        fs = reader.getSampleFrequency(i)
        seg = reader.readSignal(i, 0, int(min(reader.getNSamples()[i], fs * 60)))
        vals = np.unique(np.round(seg, 6))
        if 1 < len(vals) <= 32 and np.allclose(seg, np.round(seg), atol=1e-6):
            names.append(name)
    return names


def load(path: str, channels=None, trigger_channel=None, drop=None) -> Recording:
    r = _open(path)
    labels = r.getSignalLabels()
    rates = [float(r.getSampleFrequency(i)) for i in range(len(labels))]

    markers = []
    onsets, durations, texts = r.readAnnotations()
    for onset, dur, text in zip(onsets, durations, texts):
        markers.append(Marker(float(onset), str(text), _numeric(str(text)), float(dur)))

    exclude = set(drop or [])
    if trigger_channel:
        if trigger_channel not in labels:
            sys.exit(f"no trigger channel {trigger_channel!r}; channels: {labels}")
        ti = labels.index(trigger_channel)
        # the trigger channel wins: numeric annotations (usually its copy) are dropped,
        # text annotations (target, epoch start, comments) are kept
        markers = [m for m in markers if m.code is None]
        markers += edges_from_trigger(r.readSignal(ti), rates[ti])
        exclude.add(trigger_channel)
    markers.sort(key=lambda m: m.onset_s)

    if channels:
        missing = [c for c in channels if c not in labels]
        if missing:
            sys.exit(f"no channels {missing}; channels: {labels}")
        names = [c for c in channels if c not in exclude]
    else:
        names = [c for c in labels if c not in exclude]
    # one stream has one rate: keep the most common one, say what is left out
    fs = Counter(rates[labels.index(c)] for c in names).most_common(1)[0][0]
    other = [c for c in names if rates[labels.index(c)] != fs]
    if other:
        print(f"skipping channels with another sample rate: {other}", file=sys.stderr)
        names = [c for c in names if c not in other]
    idx = [labels.index(c) for c in names]
    scale, units = [], []
    for i in idx:
        u = _unit(r, i)
        if u.lower() in VOLT_TO_UV:
            scale.append(VOLT_TO_UV[u.lower()])
            units.append("microvolts")
        else:
            scale.append(1.0)
            units.append(u or "n/a")
    start = r.getStartdatetime()
    return Recording(path, fs, names, units, int(r.getNSamples()[idx[0]]), markers,
                     start.isoformat() if start else None, r, idx, np.array(scale))


# ───────────────────────── marker map → speller-link ─────────────────────────

class MarkerMap:
    """Translates the file's marker labels into speller-link messages.

    JSON (every section optional; see markers_TEMPLATE.json):
    {
      "symbols": ["А", …], "layout": [rows, cols],     # the keyboard shown in the recording
      "epoch_start": ["epoch*"], "epoch_end": ["end*"], # fnmatch patterns
      "flash": [                                        # regex rules, first match wins
        {"match": "^R(\\d)$", "row_from_group": 1, "offset": 1},
        {"match": "^C(\\d+)$", "col_from_group": 1, "offset": 1},
        {"match": "^S(\\d+)$", "symbol_from_group": 1, "offset": 1},
        {"match": "^flash:(.+)$", "symbols_from_group": 1, "sep": ","},
        {"match": "^X$", "symbols": ["К"]}
      ],
      "cue": {"match": "^target:(.+)$"}
    }
    A flash rule may add "epoch_group": n — the epoch number is in the label itself;
    a new number starts a new epoch.
    """

    def __init__(self, cfg: dict):
        self.symbols = list(cfg.get("symbols", []))
        self.layout = tuple(cfg.get("layout", [0, 0]))
        if self.symbols and self.layout[0] * self.layout[1] != len(self.symbols):
            raise ValueError("marker map: len(symbols) must equal rows × cols")
        self.epoch_start = cfg.get("epoch_start", [])
        self.epoch_end = cfg.get("epoch_end", [])
        self.rules = [dict(r, _re=re.compile(r["match"])) for r in cfg.get("flash", [])]
        cue = cfg.get("cue")
        self.cue_re = re.compile(cue["match"]) if cue else None

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def rows(self):
        r, c = self.layout
        return [self.symbols[i * c:(i + 1) * c] for i in range(r)]

    def classify(self, label: str):
        """→ (kind, symbols_or_cue, epoch_or_None); kind is epoch_start, epoch_end, cue, flash or None."""
        if any(fnmatch.fnmatchcase(label, p) for p in self.epoch_start):
            return "epoch_start", None, None
        if any(fnmatch.fnmatchcase(label, p) for p in self.epoch_end):
            return "epoch_end", None, None
        if self.cue_re and (m := self.cue_re.match(label)):
            return "cue", m.group(1), None
        for rule in self.rules:
            m = rule["_re"].match(label)
            if not m:
                continue
            epoch = int(m.group(rule["epoch_group"])) if "epoch_group" in rule else None
            off = rule.get("offset", 0)
            if "symbols" in rule:
                syms = list(rule["symbols"])
            elif "symbols_from_group" in rule:
                s = m.group(rule["symbols_from_group"])
                syms = s.split(rule["sep"]) if rule.get("sep") else list(s)
            elif "row_from_group" in rule:
                syms = self.rows()[int(m.group(rule["row_from_group"])) - off]
            elif "col_from_group" in rule:
                j = int(m.group(rule["col_from_group"])) - off
                syms = [row[j] for row in self.rows()]
            elif "symbol_from_group" in rule:
                syms = [self.symbols[int(m.group(rule["symbol_from_group"])) - off]]
            else:
                raise ValueError(f"flash rule {rule['match']!r} says nothing about symbols")
            return "flash", syms, epoch
        return None, None, None


# ─────────────────────────────────── LSL ─────────────────────────────────────

def eeg_outlet(ch_names, units, srate, origin, source_id, extra=None):
    info = pylsl.StreamInfo(EEG_STREAM, "EEG", len(ch_names), srate, pylsl.cf_float32, source_id)
    desc = info.desc()
    desc.append_child_value("origin", origin)              # replay | live
    for k, v in (extra or {}).items():
        desc.append_child_value(k, str(v))
    chs = desc.append_child("channels")
    for name, unit in zip(ch_names, units):
        ch = chs.append_child("channel")
        ch.append_child_value("label", name)
        ch.append_child_value("unit", unit)
        ch.append_child_value("type", "EEG" if unit == "microvolts" else "AUX")
    return pylsl.StreamOutlet(info)


def marker_outlet(name, source_id):
    info = pylsl.StreamInfo(name, "Markers", 1, pylsl.IRREGULAR_RATE, pylsl.cf_string, source_id)
    info.desc().append_child_value("protocol", PROTOCOL)
    return pylsl.StreamOutlet(info)


def _push(outlet, msg, ts):
    outlet.push_sample([json.dumps(msg, ensure_ascii=False)], ts)


# ────────────────────────────────── replay ───────────────────────────────────

class _Epochs:
    """Epoch bookkeeping for --as-stimulus / --as-control."""

    def __init__(self, mmap, link, control):
        self.mmap, self.link, self.control = mmap, link, control
        self.current, self.done, self.count = None, set(), 0

    def start(self, epoch, ts):
        self.finish(ts)
        self.current = epoch
        self.count += 1
        if self.control:
            _push(self.control, {"cmd": "epoch_start", "epoch": epoch, "symbols": self.mmap.symbols,
                                 "layout": list(self.mmap.layout), "text": "", "settings": {},
                                 "source": "replay"}, ts)

    def finish(self, ts):
        if self.current is not None and self.current not in self.done:
            self.done.add(self.current)
            if self.link:
                self.link.done(epoch=self.current)

    def flash(self, symbols, epoch, ts):
        if epoch is not None and epoch != self.current:
            self.start(epoch, ts)
        elif epoch is None and (self.current is None or self.current in self.done):
            self.start((self.current + 1) if self.current is not None else 0, ts)
        if self.link:
            self.link.flash(symbols, timestamp=ts, epoch=self.current)


def replay(rec, speed=1.0, start_s=0.0, stop_s=None, mmap=None, as_stimulus=False,
           as_control=False, wait_consumers=5.0, log=print):
    file_id = hashlib.sha1(os.path.abspath(rec.path).encode()).hexdigest()[:10]
    eeg = eeg_outlet(rec.ch_names, rec.units, rec.srate, "replay", f"speller-eeg-replay-{file_id}",
                     {"file": os.path.basename(rec.path), "speed": speed, "start_s": start_s,
                      "recorded_at": rec.start})
    raw_markers = marker_outlet(REPLAY_MARKERS_STREAM, f"speller-replay-markers-{file_id}")
    control = marker_outlet(CONTROL_STREAM, f"speller-control-replay-{file_id}") if as_control else None
    link = None
    if as_stimulus:
        link = SpellerLink(name="eeg_source_replay", role="stimulus", wait_for_speller=0,
                           keyboard=mmap.symbols, layout=mmap.layout)

    t_wait = time.monotonic()      # let the decoder (and speller) attach before the first sample
    while time.monotonic() - t_wait < wait_consumers:
        if (eeg.have_consumers() and (link is None or link._stim.have_consumers())
                and (control is None or control.have_consumers())):
            break
        time.sleep(0.05)
    if link:
        link.send_keyboard()
    epochs = _Epochs(mmap, link, control) if mmap else None

    i0 = int(round(start_s * rec.srate))
    i1 = rec.n_samples if stop_s is None else min(rec.n_samples, int(round(stop_s * rec.srate)))
    step = max(1, int(round(CHUNK_S * rec.srate)))
    markers = [m for m in rec.markers if i0 <= m.onset_s * rec.srate < i1]
    stats = {"samples": 0, "markers": 0, "flashes": 0, "epochs": 0, "late_ms_max": 0.0}

    # t0 is the LSL time of sample i0. With speed != 1 timestamps stay on the file's
    # time scale and only arrive faster: decoder windows are exact, but a live
    # speller must not be run like that.
    t0 = pylsl.local_clock() + 0.1
    wall0 = time.perf_counter() + 0.1
    log(f"replay: {os.path.basename(rec.path)} · {len(rec.ch_names)} ch · {rec.srate:g} Hz · "
        f"{(i1 - i0) / rec.srate:.1f} s · {len(markers)} markers · ×{speed:g}")
    mi = 0
    for a in range(i0, i1, step):
        b = min(a + step, i1)
        delay = wall0 + (b - i0) / rec.srate / speed - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            stats["late_ms_max"] = max(stats["late_ms_max"], -delay * 1000)
        eeg.push_chunk(rec.read(a, b), [t0 + (i - i0) / rec.srate for i in range(a, b)])
        stats["samples"] += b - a

        while mi < len(markers) and markers[mi].onset_s * rec.srate < b:
            m = markers[mi]
            mi += 1
            ts = t0 + (m.onset_s * rec.srate - i0) / rec.srate
            _push(raw_markers, {"label": m.label, "code": m.code, "onset_s": round(m.onset_s, 6),
                                "duration_s": m.duration_s}, ts)
            stats["markers"] += 1
            if not epochs:
                continue
            kind, data, epoch = mmap.classify(m.label)
            if kind == "epoch_start":
                epochs.start(0 if epochs.current is None else epochs.current + 1, ts)
            elif kind == "epoch_end":
                epochs.finish(ts)
            elif kind == "flash":
                epochs.flash(data, epoch, ts)
                stats["flashes"] += 1
            # "cue" stays in SpellerReplayMarkers only: the contract does not send targets

    if epochs:
        epochs.finish(pylsl.local_clock())
        stats["epochs"] = epochs.count
    rec.close()
    time.sleep(0.5)                # let the last windows reach the decoder
    if control:
        _push(control, {"cmd": "end", "reason": "replay finished"}, pylsl.local_clock())
        time.sleep(0.3)
    log(f"replay: done · {stats['samples']} samples · {stats['markers']} markers · "
        f"{stats['flashes']} flashes · {stats['epochs']} epochs · max lateness {stats['late_ms_max']:.1f} ms")
    return stats


# ─────────────────────────────────── live ────────────────────────────────────

def live(find, timeout=10.0, dejitter=False, max_seconds=None, log=print):
    """Relay an amplifier's LSL stream as SpellerEEG with its own timestamps.

    ``dejitter`` smooths timestamps by the nominal rate; use it only after checking
    (the tool prints the actual rate every 10 s) that the amplifier keeps its rate.
    """
    key, _, value = find.partition("=")
    log(f"live: looking for {key}={value} …")
    streams = [s for s in pylsl.resolve_byprop(key, value, timeout=timeout) if s.name() != EEG_STREAM]
    if not streams:
        sys.exit(f"live: no stream {find} within {timeout:g} s")
    flags = pylsl.proc_clocksync | (pylsl.proc_dejitter if dejitter else 0)
    inlet = pylsl.StreamInlet(streams[0], max_buflen=60, processing_flags=flags)
    full = inlet.info()
    names, units, ch = [], [], full.desc().child("channels").child("channel")
    while not ch.empty():
        names.append(ch.child_value("label"))
        units.append(ch.child_value("unit") or "microvolts")
        ch = ch.next_sibling()
    if len(names) != full.channel_count():
        names = [f"ch{i + 1}" for i in range(full.channel_count())]
        units = ["microvolts"] * full.channel_count()
    srate = full.nominal_srate()
    log(f"live: {full.name()} ({full.type()}) · {len(names)} ch · {srate:g} Hz · {full.hostname()}")
    out = eeg_outlet(names, units, srate, "live", f"speller-eeg-live-{full.source_id() or full.uid()}",
                     {"upstream": full.name(), "upstream_host": full.hostname()})
    n, gaps, last_ts = 0, 0, None
    t_start = t_report = t_data = time.monotonic()
    while max_seconds is None or time.monotonic() - t_start < max_seconds:
        data, ts = inlet.pull_chunk(timeout=0.0)   # no waiting: a timeout waits for a full chunk
        if not ts:
            if time.monotonic() - t_data > 1.0:
                log("live: no data for 1 s — is the amplifier running?")
                t_data = time.monotonic()
            time.sleep(0.001)
            continue
        t_data = time.monotonic()
        if last_ts is not None and srate > 0 and ts[0] - last_ts > 3 / srate:
            gaps += 1
            log(f"live: gap of {1000 * (ts[0] - last_ts):.0f} ms")
        out.push_chunk(np.asarray(data, np.float32), list(ts))
        n += len(ts)
        last_ts = ts[-1]
        if time.monotonic() - t_report > 10:
            log(f"live: {n / (time.monotonic() - t_report):.1f} Hz actual (nominal {srate:g}), {gaps} gaps")
            n, gaps, t_report = 0, 0, time.monotonic()


# ─────────────────────────────────── info ────────────────────────────────────

def info(path):
    r = _open(path)
    labels = r.getSignalLabels()
    print(f"file: {path}")
    print(f"start: {r.getStartdatetime()} · duration {r.getFileDuration():.1f} s · {len(labels)} channels")
    for i, name in enumerate(labels):
        print(f"  {name:16s} {r.getSampleFrequency(i):g} Hz  [{_unit(r, i) or '—'}]")
    cands = trigger_candidates(r)
    print("look like triggers:", ", ".join(cands) if cands else "none")
    onsets, _, texts = r.readAnnotations()
    texts = [str(t) for t in texts]
    print(f"EDF+ annotations: {len(texts)}")
    print("  by pattern (digits -> #):")
    for text, k in Counter(re.sub(r"\d+", "#", t) for t in texts).most_common(30):
        print(f"  {k:6d} × {text!r}")
    print("  first 15:")
    for onset, text in list(zip(onsets, texts))[:15]:
        print(f"  {onset:10.4f} s  {text!r}")


# ──────────────────────────────────── CLI ────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    pi = sub.add_parser("info", help="what is in the file")
    pi.add_argument("edf")
    pr = sub.add_parser("replay", help="play a recording into LSL in real time")
    pr.add_argument("edf")
    pr.add_argument("--speed", type=float, default=1.0, help="1 = real time (default)")
    pr.add_argument("--start", type=float, default=0.0, help="from this second of the file")
    pr.add_argument("--stop", type=float, help="up to this second of the file")
    pr.add_argument("--channels", nargs="+", help="only these channels")
    pr.add_argument("--drop", nargs="+", help="leave these channels out")
    pr.add_argument("--trigger-channel", help="markers from this channel instead of annotations")
    pr.add_argument("--marker-map", help="JSON that translates the file's markers into speller-link")
    pr.add_argument("--as-stimulus", action="store_true",
                    help="send keyboard and flashes to SpellerStimuli (instead of a stimulus module)")
    pr.add_argument("--as-control", action="store_true",
                    help="send epoch_start/end to SpellerControl (only when no speller is running)")
    pr.add_argument("--wait", type=float, default=10.0,
                    help="seconds to wait for the decoder to attach before playing (default 10)")
    pl = sub.add_parser("live", help="relay an amplifier stream as SpellerEEG")
    pl.add_argument("--find", default="type=EEG", help="e.g. type=EEG or name=MyAmplifier")
    pl.add_argument("--timeout", type=float, default=10.0)
    pl.add_argument("--dejitter", action="store_true", help="smooth the amplifier's timestamps")
    a = ap.parse_args(argv)

    if a.mode == "info":
        info(a.edf)
    elif a.mode == "replay":
        if (a.as_stimulus or a.as_control) and not a.marker_map:
            ap.error("--as-stimulus/--as-control need --marker-map")
        if a.speed != 1.0 and a.as_control:
            print("warning: with --speed != 1 a live speller would get epochs out of real time")
        mmap = MarkerMap.load(a.marker_map) if a.marker_map else None
        if a.as_stimulus and not mmap.symbols:
            ap.error("--as-stimulus needs symbols and layout in the marker map")
        rec = load(a.edf, a.channels, a.trigger_channel, a.drop)
        replay(rec, a.speed, a.start, a.stop, mmap, a.as_stimulus, a.as_control, a.wait)
    else:
        live(a.find, a.timeout, a.dejitter)


if __name__ == "__main__":
    main()
