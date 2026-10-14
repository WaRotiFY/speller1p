"""Example stimulus module: classic 6x6 matrix, rows and columns flash (pygame).

Run the speller first, then:

    python stimulus_kit/examples/pygame_matrix.py              # EEG decoder runs elsewhere
    python stimulus_kit/examples/pygame_matrix.py --keyboard   # SPACE stands in for EEG

Everything the speller needs from this program is one call after each flip:
``link.flash(symbols)``. The rest (waiting for an epoch, stopping early, showing the
result) is optional and uses the same ``link`` object.
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import pygame  # noqa: E402
from speller_link import SpellerLink, now  # noqa: E402
from keyboard_decoder import KeyboardDecoder  # noqa: E402

BG, DIM, LIT, TEXT, ACCENT = (12, 14, 20), (90, 96, 110), (255, 255, 255), (220, 224, 232), (255, 196, 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keyboard", action="store_true", help="decode SPACE presses instead of EEG")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--photodiode", action="store_true", help="white square in the bottom-left corner on every flash")
    ap.add_argument("--flash-ms", type=int, help="default: value suggested by the speller")
    ap.add_argument("--isi-ms", type=int, help="default: value suggested by the speller")
    ap.add_argument("--repetitions", type=int, help="maximum repetitions; default: suggested by the speller")
    ap.add_argument("--size", default="1000x760")
    args = ap.parse_args()

    pygame.init()
    flags = pygame.SCALED | (pygame.FULLSCREEN if args.fullscreen else 0)
    w, h = map(int, args.size.split("x"))
    screen = pygame.display.set_mode((w, h), flags, vsync=1)
    pygame.display.set_caption("Speller — matrix")
    big = pygame.font.SysFont("dejavusans,arial", h // 11)
    small = pygame.font.SysFont("dejavusans,arial", h // 22)

    link = SpellerLink(name="pygame_matrix", role="stimulus+decoder" if args.keyboard else "stimulus")
    flip_cost = []
    while True:
        epoch = wait_epoch(link, screen, small)
        if epoch is None:
            break
        cfg = epoch.settings
        flash_s = (args.flash_ms or cfg.get("flash_ms", 100)) / 1000
        isi_s = (args.isi_ms or cfg.get("isi_ms", 150)) / 1000
        reps = args.repetitions or cfg.get("n_repetitions", 5)
        rows = epoch.rows()
        groups = [r for r in rows] + [[r[c] for r in rows] for c in range(len(rows[0]))]
        decoder = KeyboardDecoder(epoch.symbols) if args.keyboard else None

        draw(screen, big, small, rows, set(), epoch.text, False)
        pygame.display.flip()
        pause(0.5)
        stopped = False
        for _ in range(reps):
            order = groups[:]
            random.shuffle(order)
            for group in order:
                draw(screen, big, small, rows, set(group), epoch.text, args.photodiode)
                t0 = time.perf_counter()
                pygame.display.flip()                 # with vsync: returns at the frame swap
                onset = link.flash(group)             # <- the one required call
                flip_cost.append(time.perf_counter() - t0)
                pressed = hold(onset + flash_s)
                draw(screen, big, small, rows, set(), epoch.text, False)
                pygame.display.flip()
                pressed |= hold(onset + flash_s + isi_s)
                if decoder:
                    decoder.add_flash(group, pressed)
                    link.send_likelihood(decoder.probabilities())
                if link.should_stop() or quit_requested():
                    stopped = True
                    break
            if stopped:
                break
        link.done()
        if decoder:
            link.send_likelihood(decoder.probabilities(), final=True)
        res = link.wait_result(timeout=5)
        show_result(screen, big, small, rows, res)
        if quit_requested():
            break
    if flip_cost:
        flip_cost.sort()
        print("flip() blocking time, ms: median %.2f, p95 %.2f" %
              (1000 * flip_cost[len(flip_cost) // 2], 1000 * flip_cost[int(len(flip_cost) * 0.95)]))
    pygame.quit()


def wait_epoch(link, screen, font):
    while True:
        epoch = link.wait_epoch(timeout=0.1)
        if epoch is not None or link.ended:
            return epoch
        if quit_requested():
            return None
        screen.fill(BG)
        msg = "Пауза" if link.paused else "Ожидание спеллера…"
        img = font.render(msg, True, DIM)
        screen.blit(img, img.get_rect(center=screen.get_rect().center))
        pygame.display.flip()


def draw(screen, big, small, rows, lit, text, photodiode):
    screen.fill(BG)
    W, H = screen.get_size()
    top = H // 6
    cw, ch = W // (len(rows[0]) + 1), (H - top) // (len(rows) + 1)
    for r, row in enumerate(rows):
        for c, sym in enumerate(row):
            label = "␣" if sym == " " else sym
            img = big.render(label, True, LIT if sym in lit else DIM)
            screen.blit(img, img.get_rect(center=(cw * (c + 1), top + ch * (r + 1))))
    t = small.render(text[-40:] + "_", True, TEXT)
    screen.blit(t, (W // 20, H // 20))
    if photodiode:
        pygame.draw.rect(screen, LIT, pygame.Rect(0, H - 60, 60, 60))


def show_result(screen, big, small, rows, res):
    if not res:
        return
    draw(screen, big, small, rows, set(), res.get("text", ""), False)
    sym = res.get("symbol")
    label = "?" if sym is None else ("␣" if sym == " " else sym)
    img = big.render(label, True, ACCENT)
    W, H = screen.get_size()
    pygame.draw.rect(screen, BG, img.get_rect(center=(W // 2, H // 2)).inflate(80, 60))
    screen.blit(img, img.get_rect(center=(W // 2, H // 2)))
    pygame.display.flip()


def hold(until):
    """Keep the window responsive until LSL time ``until``; True if SPACE was pressed."""
    pressed = False
    while now() < until:
        for ev in pygame.event.get(pygame.KEYDOWN):
            if ev.key == pygame.K_SPACE:
                pressed = True
        time.sleep(0.001)
    return pressed


def pause(seconds):
    hold(now() + seconds)


def quit_requested():
    for ev in pygame.event.get(pygame.QUIT):
        return True
    keys = pygame.key.get_pressed()
    return bool(keys[pygame.K_ESCAPE])


if __name__ == "__main__":
    main()
