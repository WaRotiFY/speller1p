"""Space bar instead of EEG: a stand-in decoder for trying a stimulus program.

The participant presses SPACE while the intended symbol is lit. Each flash yields
a likelihood update under a simple response model:

    P(press | symbol was lit)     = hit
    P(press | symbol was not lit) = false_alarm

A real EEG decoder replaces this class; the stimulus program does not change.
"""
import math


class KeyboardDecoder:
    def __init__(self, symbols, hit=0.85, false_alarm=0.05):
        self.symbols = list(symbols)
        self.hit, self.fa = hit, false_alarm
        self.logp = {s: 0.0 for s in self.symbols}
        self.flashes = 0

    def add_flash(self, lit, pressed):
        """Update with one flash: ``lit`` symbols, ``pressed`` = SPACE during this flash."""
        lit = set(lit)
        for s in self.symbols:
            if s in lit:
                p = self.hit if pressed else 1 - self.hit
            else:
                p = self.fa if pressed else 1 - self.fa
            self.logp[s] += math.log(p)
        self.flashes += 1

    def probabilities(self):
        top = max(self.logp.values())
        w = {s: math.exp(v - top) for s, v in self.logp.items()}
        total = sum(w.values())
        return {s: v / total for s, v in w.items()}
