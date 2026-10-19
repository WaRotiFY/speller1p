"""По одной букве в центре экрана (RSVP, pygame).

A completely different picture from ``pygame_matrix.py``; the speller does not
notice the difference. Run the speller first, then:

    python stimulus_kit/examples/pygame_rsvp.py --keyboard
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

KEYBOARD = list("абвгдежзийклмнопрстуфхцчшщъыьэюя_.,⌫")     # the module owns its keyboard ("_" is a space)

BG, FG, DIM, ACCENT = (8, 8, 12), (255, 255, 255), (110, 110, 120), (120, 220, 140)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keyboard", action="store_true", help="decode SPACE presses instead of EEG")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--photodiode", action="store_true")
    ap.add_argument("--flash-ms", type=int, default=150)
    ap.add_argument("--isi-ms", type=int, default=100)
    ap.add_argument("--repetitions", type=int, default=3)
    args = ap.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((800, 600), pygame.SCALED | (pygame.FULLSCREEN if args.fullscreen else 0), vsync=1)
    pygame.display.set_caption("Speller — RSVP")
    huge = pygame.font.SysFont("dejavusans,arial", 240)
    small = pygame.font.SysFont("dejavusans,arial", 30)
    link = SpellerLink(name="pygame_rsvp", role="stimulus+decoder" if args.keyboard else "stimulus",
                       keyboard=KEYBOARD, layout=(6, 6))

    while True:
        epoch = link.wait_epoch(timeout=0.2)
        if link.ended or any(e.type == pygame.QUIT for e in pygame.event.get(pygame.QUIT)):
            break
        if epoch is None:
            continue
        decoder = KeyboardDecoder(epoch.symbols) if args.keyboard else None
        frame(screen, small, epoch.text, None, huge)
        pygame.display.flip()
        wait(now() + 0.6)
        for _ in range(args.repetitions):
            order = epoch.symbols[:]
            random.shuffle(order)
            for sym in order:
                frame(screen, small, epoch.text, sym, huge, args.photodiode)
                pygame.display.flip()
                onset = link.flash([sym])               # <- the one required call
                pressed = wait(onset + args.flash_ms / 1000)
                frame(screen, small, epoch.text, None, huge)
                pygame.display.flip()
                pressed |= wait(onset + (args.flash_ms + args.isi_ms) / 1000)
                if decoder:
                    decoder.add_flash([sym], pressed)
                    link.send_likelihood(decoder.probabilities())
                if link.should_stop():
                    break
            if link.should_stop():
                break
        link.done()
        if decoder:
            link.send_likelihood(decoder.probabilities(), final=True)
        res = link.wait_result(timeout=5)
        if res:
            frame(screen, small, res.get("text", ""), res.get("symbol") or "?", huge, color=ACCENT)
            pygame.display.flip()
    pygame.quit()


def frame(screen, small, text, sym, huge, photodiode=False, color=FG):
    screen.fill(BG)
    W, H = screen.get_size()
    if sym is not None:
        img = huge.render("␣" if sym == " " else sym, True, color)
        screen.blit(img, img.get_rect(center=(W // 2, H // 2)))
    else:
        pygame.draw.line(screen, DIM, (W // 2 - 12, H // 2), (W // 2 + 12, H // 2), 2)
        pygame.draw.line(screen, DIM, (W // 2, H // 2 - 12), (W // 2, H // 2 + 12), 2)
    t = small.render(text[-30:] + "_", True, DIM)
    screen.blit(t, (20, 20))
    if photodiode:
        pygame.draw.rect(screen, FG, pygame.Rect(0, H - 50, 50, 50))


def wait(until):
    pressed = False
    while now() < until:
        for ev in pygame.event.get(pygame.KEYDOWN):
            pressed |= ev.key == pygame.K_SPACE
        time.sleep(0.001)
    return pressed


if __name__ == "__main__":
    main()
