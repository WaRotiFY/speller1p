"""Non-linguistic baselines over the actual keyboard."""
from collections import Counter
import numpy as np
from eeg_speller.core.distributions import SymbolDistribution, log_probs
from eeg_speller.core.keyboard import BACKSPACE, typed_char


class SimplePrior:
    def __init__(self, cfg):
        self.backend = cfg['backend']
        self.model_id = self.backend

    def probabilities(self, text, repertoire):
        symbols = repertoire.symbols
        if self.backend == 'uniform':
            return np.full(len(symbols), 1 / len(symbols))
        counts = Counter(text)
        groups = Counter(typed_char(s) for s in symbols if s != BACKSPACE)
        p = np.array([0 if s == BACKSPACE else counts[typed_char(s)] / groups[typed_char(s)]
                      for s in symbols], dtype=float)
        return p / p.sum() if p.sum() else np.full(len(symbols), 1 / len(symbols))

    def predict(self, text, repertoire):
        # The established numerical floor only protects log fusion; raw frequencies
        # are retained separately in prior_computed for reproducible analysis.
        return SymbolDistribution(repertoire, log_probs(self.probabilities(text, repertoire)),
                                  self.model_id, len(text))

    def top_words(self, context, n):
        return []
