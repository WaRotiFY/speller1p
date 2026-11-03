"""Проигрывание записи ЭЭГ — реальная запись вместо испытуемого, с живым спеллером.

На каждую эпоху спеллера модуль проигрывает следующий символ записи: ЭЭГ идёт в поток
SpellerEEG, вспышки строк и столбцов — в SpellerStimuli в тот момент, когда до них
доходит запись. Декодер (по умолчанию external_pipeline_decoder.py) модуль запускает
сам и, если модели ещё нет, сначала обучает её на первых символах записи. Спеллер
получает правдоподобие от декодера, складывает его с языковой моделью, может
остановить символ досрочно — тогда вспышки обрываются, как в живом эксперименте.

Окно модуля — экран испытуемого: матрица, вспышки, подсказка цели из разметки записи.
Все параметры — в settings.json рядом.
"""
from __future__ import annotations

import atexit
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "stimulus_kit"))
sys.path.insert(0, str(ROOT / "stimulus_kit" / "tools"))

import pylsl  # noqa: E402
from module_tools import color, fail, log_path, module_args, place_window  # noqa: E402
from speller_link import SpellerLink, now  # noqa: E402

DECODER = ROOT / "stimulus_kit" / "decoders" / "external_pipeline_decoder.py"
CHUNK_S = 0.02


def project_path(p) -> Path:
    """Paths in settings.json are relative to the project folder (modules run in the session folder)."""
    p = Path(str(p))
    return p if p.is_absolute() else ROOT / p


def read_epochs(rec, mmap):
    """[{"cue", "flashes": [(onset_s, symbols)]}] in file order."""
    epochs = []
    for m in rec.markers:
        kind, val, _ = mmap.classify(m.label)
        if kind == "epoch_start" or (kind == "flash" and not epochs):
            epochs.append({"cue": None, "flashes": []})
        if kind == "cue":
            epochs[-1]["cue"] = val
        elif kind == "flash":
            epochs[-1]["flashes"].append((m.onset_s, val))
    return [e for e in epochs if e["flashes"]]


def run_logged(cmd, name):
    log = open(log_path(name), "w", encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))


class Screen:
    """Participant screen: matrix, lit row/column, cue, typed text. ``None`` = no window."""

    def __init__(self, cfg, windowed, symbols, layout):
        import pygame
        self.pg = pygame
        if not windowed:
            place_window(cfg.get("window_x"), cfg.get("window_y"))
        pygame.display.init()                        # no audio: nothing to play, and no ALSA noise
        pygame.font.init()
        self.s = (pygame.display.set_mode((1000, 700)) if windowed
                  else pygame.display.set_mode((0, 0), pygame.FULLSCREEN))
        pygame.display.set_caption("Speller — проигрывание записи ЭЭГ")
        self.W, self.H = self.s.get_size()
        size = int(cfg.get("font_size", 70))
        self.font = pygame.font.SysFont(cfg.get("font_name", "Arial"), size, bold=True)
        self.small = pygame.font.SysFont(cfg.get("font_name", "Arial"), max(18, size // 3))
        self.bg = color(cfg.get("background"), (12, 14, 20))
        self.fg = color(cfg.get("letter_color"), (90, 96, 110))
        self.lit = color(cfg.get("highlight_color"), (255, 255, 255))
        self.cue_c = color(cfg.get("cue_color"), (255, 196, 60))
        self.rows = [symbols[i * layout[1]:(i + 1) * layout[1]] for i in range(layout[0])]
        self.quit = self.esc = False

    def events(self):
        for ev in self.pg.event.get():
            if ev.type == self.pg.QUIT:
                self.quit = True
            elif ev.type == self.pg.KEYDOWN and ev.key == self.pg.K_ESCAPE:
                self.esc = True

    def draw(self, lit=(), cue=None, chosen=None, text="", status=""):
        self.events()
        s = self.s
        s.fill(self.bg)
        top = self.H // 6
        cw, ch = self.W // (len(self.rows[0]) + 1), (self.H - top) // (len(self.rows) + 1)
        for r, row in enumerate(self.rows):
            for c, sym in enumerate(row):
                col = self.cue_c if sym in (cue, chosen) else (self.lit if sym in lit else self.fg)
                img = self.font.render(sym, True, col)
                s.blit(img, img.get_rect(center=(cw * (c + 1), top + ch * (r + 1))))
        s.blit(self.small.render(text[-40:] + "_", True, self.fg), (self.W // 20, self.H // 24))
        if status:
            img = self.small.render(status, True, self.fg)
            s.blit(img, img.get_rect(bottomright=(self.W - 20, self.H - 14)))
        self.pg.display.flip()

    def close(self):
        self.pg.quit()


class NoScreen:
    quit = esc = False

    def draw(self, *a, **k):
        pass

    def events(self):
        pass

    def close(self):
        pass


def main(default_settings: Path = HERE / "settings.json"):
    """Run with ``default_settings`` (other recordings: a module folder with its own settings.json)."""
    args = module_args(__doc__, str(default_settings))
    cfg = args.config
    from eeg_source import MarkerMap, eeg_outlet, load

    edf, mmap_path = project_path(cfg["edf"]), project_path(cfg["marker_map"])
    if not edf.is_file():
        fail(f"нет записи {edf}. Скачайте: uv run --no-sync python "
             f"stimulus_kit/tools/open_data/get_bnci2014_008.py")
    mmap = MarkerMap.load(str(mmap_path))
    rec = load(str(edf))
    epochs = read_epochs(rec, mmap)
    n_cal = int(cfg.get("calibration_symbols", 15))
    first = int(cfg.get("first_symbol", n_cal) if cfg.get("first_symbol") is not None else n_cal)
    print(f"запись {edf.name}: {len(epochs)} символов, {len(rec.ch_names)} кан., {rec.srate:g} Гц; "
          f"проигрываю с символа {first + 1}", flush=True)
    task = "".join(e["cue"] or "?" for e in epochs[first:])
    print(f"задание этой части записи (для сверки выберите его в окне оператора): {task}", flush=True)
    data = rec.read(0, rec.n_samples)                  # (n, ch), µV
    fs = rec.srate

    procs = []
    atexit.register(lambda: [p.terminate() for p in procs if p.poll() is None])
    if cfg.get("start_decoder", True):
        model = project_path(cfg.get("model") or f"runs/models/{edf.stem}_{cfg.get('pipeline', 'xdawn_riemann')}.joblib")
        if not model.is_file():
            model.parent.mkdir(parents=True, exist_ok=True)
            print(f"обучаю декодер на первых {n_cal} символах → {model}", flush=True)
            cal = subprocess.run([sys.executable, str(DECODER), "calibrate", str(edf), "--marker-map", str(mmap_path),
                                  "--train-epochs", str(n_cal), "--pipeline", cfg.get("pipeline", "xdawn_riemann"),
                                  "--out", str(model)], capture_output=True, text=True, encoding="utf-8")
            print(cal.stdout.strip() or cal.stderr.strip()[-600:], flush=True)
            if cal.returncode:
                fail("калибровка декодера не удалась")
        procs.append(run_logged([sys.executable, str(DECODER), "run", "--model", str(model),
                                 "--eeg-name", "SpellerEEG"], "decoder.txt"))
        print(f"декодер запущен (модель {model.name}), его вывод — logs/decoder.txt", flush=True)

    eeg = eeg_outlet(rec.ch_names, rec.units, fs, "replay", f"speller-eeg-module-{edf.stem}",
                     {"file": edf.name, "speed": cfg.get("speed", 1.0)})
    if procs:                                         # first symbol only once the decoder hears the EEG
        print("жду, пока декодер подключится к ЭЭГ…", flush=True)
        if not eeg.wait_for_consumers(float(cfg.get("decoder_wait_s", 90))):
            fail("декодер не подключился к SpellerEEG — см. logs/decoder.txt")
    link = SpellerLink(name="eeg_replay", role="stimulus", keyboard=mmap.symbols, layout=mmap.layout)
    time.sleep(1.5)                                   # let the decoder find this module's flash stream
    screen = Screen(cfg, args.windowed, mmap.symbols, mmap.layout) if cfg.get("show_window", True) else NoScreen()

    speed = float(cfg.get("speed", 1.0))
    pre, post = cfg.get("pre_ms", 500) / 1000, cfg.get("post_ms", 1000) / 1000
    flash_s = cfg.get("flash_ms", 125) / 1000
    n_groups = sum(mmap.layout)
    k = first
    last_ts = 0.0
    log = open(log_path(time.strftime("replay_%Y-%m-%d_%H-%M-%S.txt")), "w", encoding="utf-8")

    while not screen.quit:
        epoch = link.wait_epoch(timeout=0.05)
        if link.ended:
            break
        if epoch is None:
            screen.draw(status="пауза" if link.paused else "ожидание спеллера…")
            continue
        if k >= len(epochs):
            print("запись закончилась: эпоха отменена, завершите сессию", flush=True)
            link.user_stop()
            screen.draw(status="запись закончилась")
            continue
        ep = epochs[k]
        cue = ep["cue"] if cfg.get("show_cue", True) else None
        log.write(f"epoch {epoch.id} ← символ записи {k + 1} (цель {ep['cue']})\n")
        t_first, t_last = ep["flashes"][0][0], ep["flashes"][-1][0]
        a = max(0, int(round((t_first - pre) * fs)))
        b = min(rec.n_samples, int(round((t_last + post) * fs)))
        # LSL time of sample a; stays on the file's scale even if --speed > 1
        t0 = max(now() + 0.05, last_ts + 0.05)
        wall0 = time.perf_counter() + 0.05
        ts_of = lambda i: t0 + (i - a) / fs                               # noqa: E731
        fi, sent, stopped = 0, 0, False
        lit, lit_until = (), -1.0
        step = max(1, int(round(CHUNK_S * fs)))
        i = a
        while i < b:
            j = min(i + step, b)
            delay = wall0 + (j - a) / fs / speed - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            eeg.push_chunk(np.ascontiguousarray(data[i:j], np.float32), [ts_of(x) for x in range(i, j)])
            last_ts = ts_of(j - 1)
            while not stopped and fi < len(ep["flashes"]) and ep["flashes"][fi][0] * fs < j:
                onset_s, syms = ep["flashes"][fi]
                fi += 1
                if link.should_stop() or link.epoch_cancelled():
                    stopped = True                          # speller is sure: no more stimuli
                    b = min(b, int(round((ep["flashes"][fi - 2][0] if fi > 1 else onset_s) * fs + post * fs)))
                    break
                t_on = ts_of(onset_s * fs)
                link.flash(syms, timestamp=t_on)
                sent += 1
                lit, lit_until = syms, onset_s + flash_s
                if sent % n_groups == 0:
                    link.trial_done()
            if fi >= len(ep["flashes"]):
                stopped = True
            file_now = j / fs
            screen.draw(lit=lit if file_now < lit_until else (), cue=cue, text=epoch.text,
                        status=f"символ записи {k + 1}/{len(epochs)} · вспышка {sent}")
            if screen.esc:
                screen.esc = False
                link.user_stop()
            if screen.quit or link.epoch_cancelled():
                break
            i = j
        if link.epoch_cancelled() or screen.quit:
            continue                                  # cancelled: the same symbol is played again
        link.done()
        res = link.wait_result(timeout=10) or {}
        chosen = res.get("symbol")
        if chosen is not None:
            k += 1                                    # no symbol (aborted epoch): the speller retries this position
        log.write(f"epoch {epoch.id}: {sent} вспышек, выбрано {chosen!r}\n")
        log.flush()
        until = time.monotonic() + cfg.get("rest_after_ms", 1000) / 1000
        while time.monotonic() < until and not screen.quit:
            screen.draw(chosen=chosen, text=res.get("text", epoch.text), status=f"выбрано: {chosen}")
            time.sleep(0.03)
    log.close()
    screen.close()


if __name__ == "__main__":
    main()
