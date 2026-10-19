"""EEG transport adapters. Signals leave adapters in volts with explicit sample times."""
from collections import deque
from pathlib import Path
import hashlib
import json
import threading
import time
from typing import Protocol
import uuid
import numpy as np


class EEGSource(Protocol):
    """Transport boundary: physical samples in volts, time expressed in source seconds."""
    kind: str
    fs: float
    channels: list[str]
    def window(self, onset: float, interval: tuple[float, float]) -> np.ndarray | None: ...
    def status(self) -> dict: ...
    def close(self) -> None: ...


def unit_scale(unit):
    scales = {"v": 1.0, "mv": 1e-3, "uv": 1e-6, "µv": 1e-6, "μv": 1e-6, "nv": 1e-9}
    try:
        return scales[unit.strip().lower()]
    except KeyError:
        raise ValueError(f"unknown EEG unit {unit!r}; configure it explicitly") from None


def selected_indices(available, channels, aliases):
    names = [aliases.get(name, name) for name in channels]
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"EEG channels missing: {missing}; available: {available}")
    indices = [available.index(name) for name in names]
    if len(set(indices)) != len(indices):
        raise ValueError("channel aliases must not map several channels to one electrode")
    return indices


class EDFSource:
    """Recorded waveform and stimulus annotations; never reads answer/label sidecars."""
    kind = "edf_replay"

    def __init__(self, cfg, channels):
        try:
            import pyedflib
        except ImportError:
            raise ValueError("EDF support requires pyedflib: uv pip install -e .") from None
        self.path = Path(cfg["path"]).resolve()
        if not self.path.is_file():
            raise ValueError(f"EDF not found: {self.path}; generate or select a recording")
        with pyedflib.EdfReader(str(self.path)) as reader:
            available = reader.getSignalLabels()
            selected = selected_indices(available, channels, cfg.get("channel_aliases", {}))
            rates = [float(reader.getSampleFrequency(i)) for i in selected]
            if max(rates) != min(rates):
                raise ValueError("selected EDF channels must have the same sample rate")
            self.fs = rates[0]
            self.data = np.stack([reader.readSignal(i) * unit_scale(reader.getPhysicalDimension(i)) for i in selected])
            onsets, _, descriptions = reader.readAnnotations()
        self.channels = channels
        self.markers = {}
        for onset, description in zip(onsets, descriptions):
            if not description.startswith("speller/"):
                continue
            parts = description.split("/")
            if len(parts) != 6:
                raise ValueError("invalid EDF speller annotation")
            _, eid, fid, kind, index, repetition = parts
            marker = {"id": int(fid), "kind": kind, "index": int(index),
                      "repetition": int(repetition), "source_time_s": float(onset)}
            self.markers.setdefault(int(eid), []).append(marker)
        if not self.markers:
            raise ValueError("EDF has no speller stimulus annotations; raw EEG alone is insufficient")
        if sorted(self.markers) != list(range(len(self.markers))):
            raise ValueError("EDF epoch annotations must be contiguous and start at zero")
        for markers in self.markers.values():
            markers.sort(key=lambda x: x["id"])
            if [m["id"] for m in markers] != list(range(len(markers))):
                raise ValueError("EDF has duplicated or missing flash annotations")
            if any(b["source_time_s"] <= a["source_time_s"] for a, b in zip(markers, markers[1:])):
                raise ValueError("EDF flash times must increase")
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def window(self, onset, interval):
        a = round(onset * self.fs) + round(interval[0] * self.fs)
        n = round((interval[1] - interval[0]) * self.fs)
        if a < 0 or a + n > self.data.shape[1]:
            raise ValueError("EDF is missing the full EEG window around a stimulus")
        return self.data[:, a:a + n]

    def status(self):
        return {"source": self.kind, "connected": True, "channels": self.channels,
                "sample_rate_hz": self.fs, "path": str(self.path), "sha256": self.sha256,
                "recorded_epochs": len(self.markers), "label": "Синхронное воспроизведение EDF"}

    def close(self):
        pass


class LSLSource:
    """Device-independent online acquisition; selected by stream ID/channel aliases."""
    kind = "lsl"

    def __init__(self, cfg, channels):
        try:
            import pylsl
        except (ImportError, RuntimeError) as exc:
            raise ValueError(f"LSL unavailable: {exc}; install .[lsl] and liblsl") from exc
        self.lsl = pylsl
        field = cfg.get("stream_field", "source_id")
        if field not in {"source_id", "name", "type"}:
            raise ValueError("LSL stream_field must be source_id, name or type")
        streams = pylsl.resolve_byprop(field, cfg["stream_value"], timeout=cfg.get("connect_timeout_s", 5))
        if len(streams) != 1:
            raise ValueError(f"expected one matching LSL stream, found {len(streams)}")
        self.inlet = pylsl.StreamInlet(streams[0], max_buflen=60, recover=True)
        self.inlet.open_stream(timeout=cfg.get("connect_timeout_s", 5))
        info = self.inlet.info(timeout=5)
        self.fs = float(info.nominal_srate())
        if self.fs <= 0:
            raise ValueError("EEG LSL stream must have a positive nominal sample rate")
        node = info.desc().child("channels").child("channel")
        available = []
        for _ in range(info.channel_count()):
            available.append(node.child_value("label"))
            node = node.next_sibling()
        # Devices without channel metadata can supply channel_order in the profile.
        if not all(available):
            available = cfg.get("channel_order", [])
        if len(available) != info.channel_count():
            raise ValueError("LSL channel metadata missing; configure source.channel_order")
        self.selected = selected_indices(available, channels, cfg.get("channel_aliases", {}))
        self.channels = channels
        self.scale = unit_scale(cfg.get("units", "uV"))
        self.buffer = deque()
        self.saved = []
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.error = None
        self.last_sample = None
        self.dropped = 0
        # Both are monotonic clocks; pair them locally, independent of UNIX time.
        t0 = time.monotonic()
        lsl_now = pylsl.local_clock()
        t1 = time.monotonic()
        self.monotonic_to_lsl = lsl_now - (t0 + t1) / 2
        self.clock_pair_uncertainty_ms = (t1 - t0) * 500
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.marker_id = "speller-markers-" + uuid.uuid4().hex
        marker_info = pylsl.StreamInfo(cfg.get("marker_stream_name", "SpellerStimuli"), "Markers", 1,
                                      pylsl.IRREGULAR_RATE, pylsl.cf_string, self.marker_id)
        self.marker_outlet = pylsl.StreamOutlet(marker_info)
        self.thread.start()

    def _read(self):
        try:
            correction = self.inlet.time_correction(timeout=5)
            corrected_at = time.monotonic()
            while not self.closed.is_set():
                samples, stamps = self.inlet.pull_chunk(timeout=0.05, max_samples=512)
                if not stamps:
                    continue
                if time.monotonic() - corrected_at > 5:
                    correction = self.inlet.time_correction(timeout=1)
                    corrected_at = time.monotonic()
                stamps = np.asarray(stamps) + correction
                values = np.asarray(samples, dtype=float)[:, self.selected].T * self.scale
                if not np.all(np.isfinite(values)) or not np.all(np.isfinite(stamps)) or np.any(np.diff(stamps) <= 0):
                    raise ValueError("invalid EEG values or non-increasing LSL sample times")
                with self.lock:
                    if self.last_sample is not None and stamps[0] - self.last_sample > 2.5 / self.fs:
                        self.dropped += 1
                    self.last_sample = float(stamps[-1])
                    self.buffer.append((stamps, values))
                    self.saved.append((stamps, values))
                    while self.buffer and stamps[-1] - self.buffer[0][0][-1] > 60:
                        self.buffer.popleft()
        except Exception as exc:
            self.error = str(exc)

    def event_time(self, monotonic_ns):
        return monotonic_ns / 1e9 + self.monotonic_to_lsl

    def publish_marker(self, event, flash):
        payload = {"epoch_id": event.epoch_id, "flash_id": event.flash_id,
                   "kind": flash["kind"], "index": flash["index"], "indices": flash["indices"],
                   "repetition": flash["repetition"], "clock_uncertainty_ms": event.uncertainty_ms}
        self.marker_outlet.push_sample([json.dumps(payload)], timestamp=self.event_time(event.onset_ns))

    def window(self, onset, interval):
        if self.error:
            raise ValueError(f"LSL acquisition failed: {self.error}")
        with self.lock:
            if not self.buffer:
                return None
            ts = np.concatenate([x[0] for x in self.buffer])
            values = np.concatenate([x[1] for x in self.buffer], axis=1)
        a, b = onset + interval[0], onset + interval[1]
        if ts[0] > a or ts[-1] < b:
            return None
        mask = (ts >= a) & (ts <= b)
        if np.max(np.diff(ts[mask])) > 2.5 / self.fs:
            raise ValueError("EEG sample gap inside a stimulus window; epoch rejected")
        # Timestamp-based interpolation also handles fractional sample alignment.
        grid = a + np.arange(round((interval[1] - interval[0]) * self.fs)) / self.fs
        return np.stack([np.interp(grid, ts, row) for row in values])

    def status(self):
        age = None if self.last_sample is None else self.lsl.local_clock() - self.last_sample
        return {"source": self.kind, "channels": self.channels, "sample_rate_hz": self.fs,
                "connected": not self.closed.is_set() and not self.error and age is not None and age < 2,
                "signal_age_s": age, "sample_gap_count": self.dropped, "error": self.error,
                "clock_pair_uncertainty_ms": self.clock_pair_uncertainty_ms,
                "monotonic_to_lsl_s": self.monotonic_to_lsl,
                "marker_source_id": self.marker_id,
                "label": "Живой поток ЭЭГ LSL"}

    def save(self, path):
        with self.lock:
            chunks = list(self.saved)
        if chunks:
            np.savez_compressed(path, timestamps_s=np.concatenate([x[0] for x in chunks]),
                                eeg_v=np.concatenate([x[1] for x in chunks], axis=1),
                                channels=self.channels, sample_rate_hz=self.fs)

    def close(self):
        self.closed.set()
        self.thread.join(timeout=6)
        self.inlet.close_stream()
        self.marker_outlet = None
