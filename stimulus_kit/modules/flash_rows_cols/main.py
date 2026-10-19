"""Мигание строк и столбцов — сменный модуль стимуляции для спеллера.

В течение одного Trial по очереди загорается каждая строка и каждый столбец
клавиатуры; Trial повторяется n_trials раз или пока спеллер не пришлёт stop.
Параметры — в settings.json рядом.
Буквы и раскладку присылает спеллер.
"""
import os
import random
import time

import pygame

from module_tools import color, load_settings, log_path, module_args, place_window
from speller_link import SpellerLink, now

HERE = os.path.dirname(os.path.abspath(__file__))


def flash_order(groups, n_trials, min_gap, rng):
    """Shuffled rows/columns for every trial, so that two flashes containing the same
    symbol are at least ``min_gap`` other flashes apart (target-agnostic version of
    "minimum non-target flashes between targets")."""
    order = []
    for _ in range(n_trials):
        for _attempt in range(200):
            trial = groups[:]
            rng.shuffle(trial)
            seq = order + trial
            ok = True
            for k in range(len(order), len(seq)):
                for back in range(1, min_gap + 1):
                    if k - back >= 0 and set(seq[k]) & set(seq[k - back]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                break
        order += trial
    return order


def main():
    args = module_args(__doc__, os.path.join(HERE, "settings.json"))
    cfg = load_settings(args.settings)
    if not args.windowed:
        place_window(cfg.get("window_x"), cfg.get("window_y"))
    pygame.init()
    if args.windowed:
        screen = pygame.display.set_mode((1100, 700), vsync=1)
    else:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, vsync=1)
    pygame.display.set_caption("Speller — flashing rows and columns")
    W, H = screen.get_size()
    fps = int(cfg.get("FPS", 60))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(cfg.get("font_name", "Arial"), int(cfg.get("font_size", 80)), bold=True)
    small = pygame.font.SysFont(cfg.get("font_name", "Arial"), max(18, int(cfg.get("font_size", 80)) // 3))
    bg = color(cfg.get("background"), (12, 14, 20))
    fg = color(cfg.get("letter_color"), (90, 96, 110))
    lit_color = color(cfg.get("highlight_color"), (255, 255, 255))
    cue_color = color(cfg.get("cue_color"), (255, 196, 60))
    rng = random.Random(cfg.get("seed"))
    log = open(log_path(time.strftime("flash_%Y-%m-%d_%H-%M-%S.txt")), "w", encoding="utf-8")

    link = SpellerLink(name="flash_rows_cols", role="stimulus+decoder" if args.keyboard else "stimulus")
    if args.keyboard:
        from keyboard_decoder import KeyboardDecoder

    state = {"quit": False}

    def events():
        pressed = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                state["quit"] = True
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                pressed = True
        return pressed

    def draw(rows, lit=(), cue=None, text=""):
        screen.fill(bg)
        top = H // 7
        cw, ch = W // (len(rows[0]) + 1), (H - top) // (len(rows) + 1)
        for r, row in enumerate(rows):
            for c, sym in enumerate(row):
                label = "_" if sym == " " else sym
                col = cue_color if sym == cue else (lit_color if sym in lit else fg)
                img = font.render(label, True, col)
                screen.blit(img, img.get_rect(center=(cw * (c + 1), top + ch * (r + 1))))
        if cfg.get("show_text", True):
            screen.blit(small.render(text[-40:] + "_", True, fg), (W // 20, H // 24))

    def hold(until, rows, **kw):
        """Keep drawing until LSL time ``until``; True if SPACE was pressed meanwhile."""
        pressed = False
        while now() < until and not state["quit"]:
            pressed |= events()
            if kw:
                draw(rows, **kw)
                pygame.display.flip()
            clock.tick(fps)
        return pressed

    while not state["quit"]:
        epoch = link.wait_epoch(timeout=0.05)
        if link.ended:
            break
        if epoch is None:
            events()
            screen.fill(bg)
            msg = small.render("Пауза" if link.paused else "Ожидание спеллера…", True, fg)
            screen.blit(msg, msg.get_rect(center=(W // 2, H // 2)))
            pygame.display.flip()
            clock.tick(30)
            continue
        rows = epoch.rows()
        n_rows, n_cols = len(rows), max(len(r) for r in rows)
        groups = [r for r in rows] + [[r[c] for r in rows if c < len(r)] for c in range(n_cols)]
        flash_s = cfg.get("flash_ms", 100) / 1000
        isi_s = cfg.get("isi_ms", 75) / 1000
        n_trials = int(cfg.get("n_trials", 5))
        decoder = KeyboardDecoder(epoch.symbols) if args.keyboard else None
        log.write(f"epoch {epoch.id} cue {epoch.cue}\n")

        hold(now() + cfg.get("rest_before_ms", 1000) / 1000, rows, text=epoch.text)
        if epoch.cue:
            hold(now() + cfg.get("cue_ms", 1000) / 1000, rows, cue=epoch.cue, text=epoch.text)
        draw(rows, text=epoch.text)
        pygame.display.flip()

        for group in flash_order(groups, n_trials, int(cfg.get("min_gap", 1)), rng):
            draw(rows, lit=group, text=epoch.text)
            pygame.display.flip()                 # with vsync: returns at the frame swap
            onset = link.flash(group)             # <- the one required call
            log.write(f"flash {''.join(group)} {onset:.6f}\n")
            pressed = hold(onset + flash_s, rows)
            draw(rows, text=epoch.text)
            pygame.display.flip()
            pressed |= hold(onset + flash_s + isi_s, rows)
            if decoder:
                decoder.add_flash(group, pressed)
                link.send_likelihood(decoder.probabilities())
            if link.should_stop() or state["quit"]:
                break
        link.done()
        if decoder:
            link.send_likelihood(decoder.probabilities(), final=True)
        res = link.wait_result(timeout=5)
        chosen = res.get("symbol") if res else None
        log.write(f"epoch {epoch.id} result {chosen}\n")
        hold(now() + cfg.get("rest_after_ms", 1000) / 1000, rows, cue=chosen,
             text=(res or {}).get("text", epoch.text))
    log.close()
    pygame.quit()


if __name__ == "__main__":
    main()
