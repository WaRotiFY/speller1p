"""Session view: the participant's screen and the experimenter's screen, side by side, live.

Only listens to the speller's LSL streams, never sends anything, so it can sit next
to a replay or a live session without changing it:

    SpellerEEG            raw EEG (scrolling traces, running ERP average)
    SpellerStimuli        keyboard and flashes (lit rows/columns on the participant screen)
    SpellerLikelihood     decoder output (heat map and top-5 on the experimenter screen)
    SpellerControl        epoch_start / result / end
    SpellerReplayMarkers  target:X of a replay — shown to the experimenter only

Not part of the speller and not in the requirements: a demonstration tool.

    python stimulus_kit/tools/open_data/session_view.py
    python stimulus_kit/tools/open_data/session_view.py --record demo.mp4 --fps 25     # also write a video
    SDL_VIDEODRIVER=dummy python …/session_view.py --record demo.mp4                    # headless
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import deque

import numpy as np
import pylsl
import pygame
from scipy.signal import butter, sosfiltfilt

W, H = 1600, 900
SPLIT = 640                          # participant screen | experimenter screen
FLASH_S = 0.125                      # how long a flashed group stays lit
EEG_SPAN_S = 5.0                     # experimenter: seconds of scrolling EEG
ERP_WIN = (-0.1, 0.8)
LANE_UV = 20.0                       # µV that fill half a lane
DISPLAY_BAND = (0.5, 12.0)           # display only; the decoder filters on its own
PAD_S = 1.0

BG_SUBJ = (12, 12, 16)
BG_OP = (24, 26, 32)
PANEL = (34, 37, 45)
GRID = (55, 58, 68)
TXT = (225, 228, 235)
DIM = (120, 125, 140)
LIT = (255, 255, 255)
ACCENT = (255, 196, 61)
GOOD = (88, 201, 120)
BAD = (235, 92, 92)
TRACE = (126, 178, 255)
TGT = (255, 140, 80)
NTG = (130, 140, 160)


def font(size, bold=False, mono=False):
    names = "dejavusansmono" if mono else ("dejavusans,arial,liberationsans")
    return pygame.font.SysFont(names, size, bold=bold)


class Streams:
    """Attaches to each named stream whenever it appears; pulls everything without blocking."""

    NAMES = ["SpellerEEG", "SpellerStimuli", "SpellerLikelihood", "SpellerControl", "SpellerReplayMarkers"]

    def __init__(self):
        self.resolvers = {n: pylsl.ContinuousResolver("name", n) for n in self.NAMES}
        self.inlets = {n: {} for n in self.NAMES}
        self.eeg_info = None
        self.last_resolve = 0.0

    def poll(self):
        if time.monotonic() - self.last_resolve > 0.5:
            self.last_resolve = time.monotonic()
            for n, r in self.resolvers.items():
                for info in r.results():
                    if info.source_id() in self.inlets[n]:
                        continue
                    inlet = pylsl.StreamInlet(info, max_buflen=30, processing_flags=pylsl.proc_clocksync)
                    try:
                        inlet.open_stream(timeout=1)
                    except Exception:
                        continue
                    self.inlets[n][info.source_id()] = inlet
                    if n == "SpellerEEG":
                        self.eeg_info = inlet.info(timeout=2)
        out = {n: [] for n in self.NAMES}
        for n, group in self.inlets.items():
            for sid, inlet in list(group.items()):
                try:
                    data, ts = inlet.pull_chunk(timeout=0.0, max_samples=4096)
                except Exception:
                    group.pop(sid, None)
                    continue
                if ts:
                    out[n].append((np.asarray(data) if n == "SpellerEEG" else data, list(ts)))
        return out


class Session:
    def __init__(self):
        self.symbols, self.layout = [], (6, 6)
        self.channels, self.fs, self.origin = [], 0.0, "?"
        self.eeg_t = deque()                     # (ts array, values (n, ch)) chunks
        self.now = None                          # latest EEG time — the display clock
        self.flashes = []                        # (ts, symbols, epoch)
        self.epoch = None
        self.epoch_flashes = 0
        self.cues = []                           # targets of a replay, in epoch order
        self.p = {}                              # current epoch: symbol -> probability
        self.info = {}
        self.typed, self.wanted, self.marks = "", "", []
        self.result_flash = None                 # (symbol, ok, wall time)
        self.ended_at = None
        self.speller_results = False             # a real speller is sending "result"
        self.finished = set()
        self.erp = {"T": [None, 0], "N": [None, 0]}
        self.pending_erp = deque()               # flashes waiting for post-stimulus EEG

    # ── inputs ──
    def eeg(self, data, ts):
        self.eeg_t.append((np.asarray(ts), data))
        self.now = ts[-1]
        while self.eeg_t and self.now - self.eeg_t[0][0][-1] > max(EEG_SPAN_S, 3) + 2:
            self.eeg_t.popleft()

    def stimulus(self, msg, ts):
        if msg.get("event") == "keyboard":
            self.symbols, self.layout = list(msg["symbols"]), tuple(msg["layout"])
        elif "flash" in msg:
            e = msg.get("epoch")
            if e != self.epoch:
                self.new_epoch(e)
            self.flashes.append((ts, msg["flash"], e))
            self.flashes = self.flashes[-400:]
            self.epoch_flashes += 1
            self.pending_erp.append((ts, msg["flash"], e))

    def control(self, msg, ts):
        cmd = msg.get("cmd")
        if cmd == "epoch_start":
            if msg.get("symbols"):
                self.symbols, self.layout = list(msg["symbols"]), tuple(msg["layout"])
            if msg["epoch"] != self.epoch:
                self.new_epoch(msg["epoch"])
        elif cmd == "result":
            self.speller_results = True
            self.finish(msg.get("symbol", "?"), msg.get("epoch"), text=msg.get("text"))
        elif cmd == "end":
            self.ended_at = time.monotonic()

    def likelihood(self, msg, ts):
        p = msg.get("p") or {}
        if not isinstance(p, dict) or not p:
            return
        tot = sum(p.values()) or 1.0
        if msg.get("epoch") == self.epoch:
            self.p = {k: v / tot for k, v in p.items()}
            self.info = msg.get("info", {})
        if msg.get("final") and not self.speller_results:
            self.finish(max(p, key=p.get), msg.get("epoch"))

    def replay_marker(self, msg, ts):
        m = re.match(r"^target:(.+)$", str(msg.get("label", "")))
        if m:
            self.cues.append(m.group(1))

    # ── state ──
    def new_epoch(self, e):
        self.epoch, self.epoch_flashes, self.p, self.info = e, 0, {}, {}

    def finish(self, symbol, epoch, text=None):
        if epoch in self.finished:
            return
        self.finished.add(epoch)
        k = len(self.marks)
        want = self.cue_for(epoch if epoch is not None else k)
        ok = None if want is None else symbol == want
        self.typed = text if text is not None else self.typed + symbol
        self.marks.append(ok)
        self.result_flash = (symbol, ok, time.monotonic())

    def cue_for(self, epoch):
        return self.cues[epoch] if epoch is not None and 0 <= epoch < len(self.cues) else None

    def target(self):
        return self.cue_for(self.epoch)

    def window(self, a, b):
        if not self.eeg_t:
            return None, None
        ts = np.concatenate([c[0] for c in self.eeg_t])
        x = np.concatenate([c[1] for c in self.eeg_t])
        m = (ts >= a) & (ts <= b)
        return ts[m], x[m]

    def update_erp(self):
        """Running average of target / non-target responses (needs the target, i.e. a replay)."""
        if self.now is None or not self.channels:
            return
        while self.pending_erp and self.pending_erp[0][0] + ERP_WIN[1] + 0.3 <= self.now:
            t0, syms, e = self.pending_erp.popleft()
            want = self.cue_for(e)
            if want is None:
                continue
            ts, x = self.window(t0 + ERP_WIN[0] - PAD_S, t0 + ERP_WIN[1] + 0.3)
            n = int(round((ERP_WIN[1] - ERP_WIN[0]) * self.fs))
            a = int(round(PAD_S * self.fs))
            if ts is None or len(ts) < a + n + 10:
                continue
            ch = self.erp_channels()
            seg = self.filt(x[:, ch].mean(axis=1))[a:a + n]
            seg = seg - seg[: int(-ERP_WIN[0] * self.fs)].mean()
            key = "T" if want in syms else "N"
            acc, k = self.erp[key]
            self.erp[key] = [seg if acc is None else acc + seg, k + 1]

    def filt(self, x):
        if not hasattr(self, "_sos"):
            self._sos = butter(2, DISPLAY_BAND, btype="bandpass", fs=self.fs, output="sos")
        return sosfiltfilt(self._sos, x, axis=0)

    def erp_channels(self):
        want = [i for i, c in enumerate(self.channels) if c.lower() in ("pz", "cz", "p3", "p4")]
        return want or list(range(min(3, len(self.channels))))


# ─────────────────────────────── drawing ───────────────────────────────

class View:
    def __init__(self, screen):
        self.s = screen
        self.f_big = font(46, bold=True)
        self.f_mid = font(24)
        self.f_mid_b = font(24, bold=True)
        self.f_small = font(16)
        self.f_tiny = font(13)
        self.f_mono = font(26, mono=True)
        self.f_hm = font(22, bold=True)

    def text(self, f, s, color, pos, anchor="topleft"):
        img = f.render(s, True, color)
        r = img.get_rect(**{anchor: pos})
        self.s.blit(img, r)
        return r

    # participant
    def subject(self, ss: Session):
        s = self.s
        pygame.draw.rect(s, BG_SUBJ, (0, 0, SPLIT, H))
        self.text(self.f_small, "ЭКРАН ИСПЫТУЕМОГО", DIM, (24, 18))
        # typed text line
        pygame.draw.rect(s, (28, 30, 38), (24, 50, SPLIT - 48, 64), border_radius=8)
        shown = ss.typed[-18:]
        self.text(self.f_mono, shown + ("▏" if int(time.monotonic() * 2) % 2 else " "), TXT, (40, 66))
        tgt = ss.target()
        if tgt:
            self.text(self.f_small, "задание: смотрите на", DIM, (24, 128))
            self.text(self.f_mid_b, tgt, ACCENT, (210, 122))
        rows, cols = ss.layout
        if not ss.symbols:
            self.text(self.f_mid, "ждём клавиатуру…", DIM, (SPLIT // 2, H // 2), "center")
            return
        lit = set()
        if ss.now is not None:
            for ts, syms, _ in reversed(ss.flashes):
                if ts <= ss.now < ts + FLASH_S:
                    lit |= set(syms)
                if ts < ss.now - 1:
                    break
        top, left = 190, 40
        cw = (SPLIT - 2 * left) / cols
        ch = min(cw, (H - top - 140) / rows)
        res = ss.result_flash if ss.result_flash and time.monotonic() - ss.result_flash[2] < 1.5 else None
        for i, sym in enumerate(ss.symbols):
            r, c = divmod(i, cols)
            cx, cy = left + cw * (c + 0.5), top + ch * (r + 0.5)
            if sym in lit:
                pygame.draw.rect(s, (60, 64, 80), (cx - cw / 2 + 4, cy - ch / 2 + 4, cw - 8, ch - 8), border_radius=10)
                self.text(self.f_big, sym, LIT, (cx, cy), "center")
            elif res and sym == res[0]:
                col = GOOD if res[1] is not False else BAD
                pygame.draw.rect(s, col, (cx - cw / 2 + 4, cy - ch / 2 + 4, cw - 8, ch - 8), 3, border_radius=10)
                self.text(self.f_big, sym, col, (cx, cy), "center")
            else:
                self.text(self.f_big, sym, (88, 92, 108), (cx, cy), "center")
        n_groups = rows + cols
        rep = ss.epoch_flashes / n_groups if n_groups else 0
        self.text(self.f_small, f"символ {0 if ss.epoch is None else ss.epoch + 1} · повторение {min(10, int(rep) + 1)}",
                  DIM, (24, H - 40))

    # experimenter
    def operator(self, ss: Session, speed_note):
        s = self.s
        x0 = SPLIT
        pygame.draw.rect(s, BG_OP, (x0, 0, W - x0, H))
        pygame.draw.line(s, (70, 74, 88), (x0, 0), (x0, H), 2)
        self.text(self.f_small, "ЭКРАН ЭКСПЕРИМЕНТАТОРА", DIM, (x0 + 24, 18))
        src = f"ЭЭГ: {ss.origin}{speed_note} · {len(ss.channels)} кан. · {ss.fs:g} Гц" if ss.fs else "ЭЭГ: ждём поток…"
        self.text(self.f_small, src, DIM, (W - 24, 18), "topright")
        self.eeg_panel(ss, x0 + 24, 48, W - x0 - 48, 360)
        self.heatmap(ss, x0 + 24, 428, 300, 300)
        self.top5(ss, x0 + 344, 428, 250, 300)
        self.erp_panel(ss, x0 + 614, 428, W - x0 - 638, 300)
        self.text_panel(ss, x0 + 24, 748, W - x0 - 48, 128)

    def eeg_panel(self, ss, x, y, w, h):
        s = self.s
        pygame.draw.rect(s, PANEL, (x, y, w, h), border_radius=8)
        self.text(self.f_tiny, "ЭЭГ (для глаз 0,5–12 Гц), мкВ · вертикальные линии — вспышки (оранжевые — с целевой буквой)", DIM, (x + 10, y + 6))
        if ss.now is None or not ss.channels:
            return
        a = ss.now - EEG_SPAN_S
        ts, xs = ss.window(a - PAD_S, ss.now)
        if ts is None or len(ts) < 64:
            return
        xs = ss.filt(xs)
        keep = ts >= a
        ts, xs = ts[keep], xs[keep]
        lx, lw = x + 54, w - 64
        n = len(ss.channels)
        lane = (h - 30) / n
        for t0, syms, e in ss.flashes:
            if a <= t0 <= ss.now:
                px = lx + (t0 - a) / EEG_SPAN_S * lw
                is_t = ss.cue_for(e) is not None and ss.cue_for(e) in syms
                pygame.draw.line(s, TGT if is_t else (64, 68, 82), (px, y + 24), (px, y + h - 6), 2 if is_t else 1)
        step = max(1, len(ts) // lw)
        px = lx + (ts[::step] - a) / EEG_SPAN_S * lw
        for j, name in enumerate(ss.channels):
            cy = y + 26 + lane * (j + 0.5)
            v = xs[::step, j]
            v = v - np.mean(v)
            py = cy - np.clip(v / LANE_UV, -1.2, 1.2) * lane * 0.45
            self.text(self.f_tiny, name, DIM, (x + 10, cy - 7))
            if len(px) > 1:
                pygame.draw.lines(s, TRACE, False, list(zip(px.tolist(), py.tolist())), 1)

    def heatmap(self, ss, x, y, w, h):
        s = self.s
        pygame.draw.rect(s, PANEL, (x, y, w, h), border_radius=8)
        self.text(self.f_tiny, "P(буква | ЭЭГ) в текущем символе", DIM, (x + 10, y + 6))
        if not ss.symbols:
            return
        rows, cols = ss.layout
        top = y + 28
        cw, chh = (w - 20) / cols, (h - 38) / rows
        pmax = max(ss.p.values()) if ss.p else 0
        tgt = ss.target()
        for i, sym in enumerate(ss.symbols):
            r, c = divmod(i, cols)
            v = ss.p.get(sym, 0.0)
            k = 0 if pmax <= 0 else (v / pmax) ** 0.6
            col = (int(40 + 215 * k), int(44 + 150 * k), int(58 - 20 * k))
            rect = (x + 10 + cw * c + 2, top + chh * r + 2, cw - 4, chh - 4)
            pygame.draw.rect(s, col, rect, border_radius=5)
            if sym == tgt:
                pygame.draw.rect(s, GOOD, rect, 2, border_radius=5)
            self.text(self.f_hm, sym, (20, 20, 24) if k > 0.55 else TXT, (rect[0] + rect[2] / 2, rect[1] + rect[3] / 2), "center")

    def top5(self, ss, x, y, w, h):
        s = self.s
        pygame.draw.rect(s, PANEL, (x, y, w, h), border_radius=8)
        self.text(self.f_tiny, "лидеры", DIM, (x + 10, y + 6))
        rows, cols = ss.layout
        n_groups = rows + cols
        scored = ss.info.get("flashes_scored", 0)
        self.text(self.f_small, f"оценено вспышек: {scored} / {ss.epoch_flashes}", TXT, (x + 10, y + 28))
        self.text(self.f_small, f"повторений: {scored // n_groups if n_groups else 0} из 10", TXT, (x + 10, y + 50))
        tgt = ss.target()
        for i, (sym, v) in enumerate(sorted(ss.p.items(), key=lambda kv: -kv[1])[:5]):
            yy = y + 92 + i * 40
            col = GOOD if sym == tgt else ACCENT
            self.text(self.f_mid_b, sym, col, (x + 12, yy))
            pygame.draw.rect(s, GRID, (x + 50, yy + 6, w - 120, 18), border_radius=4)
            pygame.draw.rect(s, col, (x + 50, yy + 6, max(2, (w - 120) * v), 18), border_radius=4)
            self.text(self.f_small, f"{v:.2f}", TXT, (x + w - 12, yy + 4), "topright")

    def erp_panel(self, ss, x, y, w, h):
        s = self.s
        pygame.draw.rect(s, PANEL, (x, y, w, h), border_radius=8)
        self.text(self.f_tiny, "средний ответ (Cz/Pz/P3/P4), мкВ", DIM, (x + 10, y + 6))
        (tsum, tn), (nsum, nn) = ss.erp["T"], ss.erp["N"]
        lx, lw, cy, sc = x + 16, w - 32, y + h / 2 + 10, (h - 70) / 2 / 5.0     # ±5 µV
        pygame.draw.line(s, GRID, (lx, cy), (lx + lw, cy))
        zx = lx + (-ERP_WIN[0]) / (ERP_WIN[1] - ERP_WIN[0]) * lw
        pygame.draw.line(s, GRID, (zx, y + 30), (zx, y + h - 30))
        for ms in (300, 500):
            px = lx + (ms / 1000 - ERP_WIN[0]) / (ERP_WIN[1] - ERP_WIN[0]) * lw
            pygame.draw.line(s, (48, 51, 60), (px, y + 30), (px, y + h - 30))
            self.text(self.f_tiny, f"{ms} мс", DIM, (px, y + h - 26), "midtop")
        for (acc, k), col, label, dy in (((nsum, nn), NTG, "не цель", 0), ((tsum, tn), TGT, "цель", 18)):
            self.text(self.f_tiny, f"{label}: {k}", col, (x + w - 10, y + 6 + dy), "topright")
            if acc is None or k == 0:
                continue
            v = acc / k
            px = lx + np.arange(len(v)) / max(1, len(v) - 1) * lw
            py = cy - np.clip(v, -6, 6) * sc
            pygame.draw.lines(s, col, False, list(zip(px.tolist(), py.tolist())), 3 if col == TGT else 2)
        if ss.erp["T"][1] == 0:
            self.text(self.f_tiny, "нужна разметка цели (replay)", DIM, (x + w / 2, cy - 30), "center")

    def text_panel(self, ss, x, y, w, h):
        s = self.s
        pygame.draw.rect(s, PANEL, (x, y, w, h), border_radius=8)
        self.text(self.f_tiny, "набрано / задание", DIM, (x + 10, y + 6))
        xx = x + 14
        for ch, ok in zip(ss.typed[-34:], ss.marks[-34:]):
            col = TXT if ok is None else (GOOD if ok else BAD)
            xx = self.text(self.f_mono, ch, col, (xx, y + 30)).right + 2
        if ss.cues:
            want = "".join(ss.cues[max(0, len(ss.marks) - 34): len(ss.marks) + 1 + max(0, 34 - len(ss.marks))])
            self.text(self.f_mono, want[:34], DIM, (x + 14, y + 70))
        known = [m for m in ss.marks if m is not None]
        if known:
            self.text(self.f_mid_b, f"точность {sum(known)}/{len(known)}", TXT, (x + w - 14, y + 34), "topright")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--record", help="also write the window to this video file (needs ffmpeg)")
    ap.add_argument("--max-seconds", type=float, help="stop after this long")
    ap.add_argument("--after-end", type=float, default=4.0, help="seconds to keep drawing after 'end'")
    a = ap.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("ЭЭГ-спеллер · испытуемый | экспериментатор")
    view, ss, streams = View(screen), Session(), Streams()
    rec = None
    if a.record:
        rec = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264",
                                "-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p", a.record],
                               stdin=subprocess.PIPE)
    clock = pygame.time.Clock()
    t_start = time.monotonic()
    speed_note = ""
    written = 0
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
        got = streams.poll()
        if streams.eeg_info is not None and not ss.channels:
            info = streams.eeg_info
            ss.fs = info.nominal_srate()
            node = info.desc().child("channels").child("channel")
            names = []
            for _ in range(info.channel_count()):
                names.append(node.child_value("label") or f"ch{len(names) + 1}")
                node = node.next_sibling()
            ss.channels = names
            ss.origin = info.desc().child_value("origin") or "live"
            sp = info.desc().child_value("speed")
            speed_note = f" ×{float(sp):g}" if sp and float(sp) != 1 else ""
        for name, handler in (("SpellerStimuli", ss.stimulus), ("SpellerControl", ss.control),
                              ("SpellerReplayMarkers", ss.replay_marker), ("SpellerLikelihood", ss.likelihood)):
            for data, ts in got[name]:
                for row, t in zip(data, ts):
                    try:
                        handler(json.loads(row[0]), t)
                    except (ValueError, KeyError, TypeError):
                        pass
        for data, ts in got["SpellerEEG"]:
            if ss.channels:
                ss.eeg(data[:, : len(ss.channels)], ts)
        ss.update_erp()
        view.subject(ss)
        view.operator(ss, speed_note)
        pygame.display.flip()
        if rec:                                  # keep the video on wall-clock time
            frame = pygame.image.tobytes(screen, "RGB")
            due = int((time.monotonic() - t_start) * a.fps) + 1
            while written < due:
                rec.stdin.write(frame)
                written += 1
        if ss.ended_at and time.monotonic() - ss.ended_at > a.after_end:
            running = False
        if a.max_seconds and time.monotonic() - t_start > a.max_seconds:
            running = False
        clock.tick(a.fps)
    if rec:
        rec.stdin.close()
        rec.wait()
    pygame.quit()


if __name__ == "__main__":
    main()
