"""Any causal language model from Hugging Face as the fast model (by name, no code).

``fast_llm.model_id`` is a Hugging Face name (``Qwen/Qwen2.5-0.5B``) or a local folder.
With ``fast_llm.allow_download: true`` the weights are downloaded on first use into the
usual Hugging Face cache; otherwise only already-downloaded models are used.

The character distribution is built from the full next-token distribution (MA-09): the
unfinished word is cut off ("token healing"), a short beam over continuations that agree
with it is expanded (all beams of one step in one batched forward pass), and the mass of
each continuation goes to the character right after the unfinished word.
"""
from __future__ import annotations

import hashlib
import re

from eeg_speller.core.distributions import SymbolDistribution, log_probs
from eeg_speller.core.keyboard import LM_ALPHABET, keyboard_text, project
from eeg_speller.llm.char_marginal import marginalize_token_paths


class HFCausal:
    """Source Model (Fast LLM); NEED-F-06; MA-08/09; Q-13."""
    version = "transformers"

    def __init__(self, cfg):
        # IMPROV[Q-13]
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ValueError('не установлены пакеты transformers и torch: uv pip install -e ".[hf]"') from exc
        self.torch = torch
        self.cfg = cfg
        self.model_id = str(cfg["model_id"]).strip()
        if not self.model_id:
            raise ValueError("enter a Hugging Face model name, e.g. Qwen/Qwen2.5-0.5B")
        local_only = not bool(cfg.get("allow_download", False))
        device = cfg.get("device", "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, local_files_only=local_only)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, local_files_only=local_only)
        except OSError as exc:
            hint = "" if not local_only else " (not downloaded yet: allow download in the operator window)"
            raise ValueError(f"cannot load {self.model_id}{hint}: {exc}") from exc
        self.model.to(device).eval()
        self.device = device
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        revision = getattr(self.model.config, "_commit_hash", None) or ""
        self.weights_hash = hashlib.sha256(f"{self.model_id}@{revision}".encode()).hexdigest()
        self.revision = revision

    def _next_probs(self, batch_ids):
        """Next-token probabilities for a batch of equally long token sequences."""
        with self.torch.no_grad():
            ids = self.torch.tensor(batch_ids, device=self.device)
            logits = self.model(ids).logits[:, -1, :].float()
            return self.torch.softmax(logits, dim=-1)

    def _beam_paths(self, context):
        text = keyboard_text(context)
        match = re.search(r"[а-я]+$", text)
        prefix = match.group() if match else ""
        healed = text[:-len(prefix)] if prefix else text
        base = self.tokenizer(healed or " ", add_special_tokens=False).input_ids or [self.tokenizer.eos_token_id]
        width = int(self.cfg["char_marginal"]["beam"])
        depth = int(self.cfg["char_marginal"]["depth"])
        beams = [(list(base), "", 1.0)]
        # Expand until every beam has passed the unfinished word: long words take several
        # tokens, so the number of steps grows with the prefix (capped).
        for _ in range(depth + len(prefix)):
            done = [b for b in beams if len(b[1]) > len(prefix)]
            todo = [b for b in beams if len(b[1]) <= len(prefix)]
            if not todo:
                break
            probs = self._next_probs([ids for ids, _, _ in todo])
            expanded = list(done)
            for row, (ids, decoded, mass) in zip(probs, todo):
                top = self.torch.topk(row, min(width, row.shape[-1]))
                for index, prob in zip(top.indices.tolist(), top.values.tolist()):
                    piece = self.tokenizer.decode([index], skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
                    cont = keyboard_text((decoded + piece).replace("\n", " ")) if piece else ""
                    if not cont or (prefix and not (prefix.startswith(cont) or cont.startswith(prefix))):
                        continue
                    expanded.append((ids + [index], cont, mass * prob))
            if not expanded:
                break
            expanded.sort(key=lambda item: -item[2])
            beams = expanded[:width]
            # Stop early once every beam already decides the next character.
            if all(len(d) > len(prefix) for _, d, _ in beams):
                break
        return prefix, [(decoded, mass) for _, decoded, mass in beams]

    def predict(self, context, repertoire):
        # ASSUMPTION[MA-09]
        # This adapter requires full vocabulary logits; it never uses top-k API logprobs.
        prefix, paths = self._beam_paths(context)
        lm = marginalize_token_paths(paths, prefix, tuple(LM_ALPHABET) + ("⌫",), 0.0)[:-1]
        p = project(lm, repertoire.symbols, float(self.cfg["char_marginal"]["backspace_prob"]),
                    float(self.cfg.get("other_prob", 0.005)))
        return SymbolDistribution(repertoire, log_probs(p), f"fast_llm:{self.model_id}", len(context))

    def top_words(self, context, n):
        from collections import defaultdict
        prefix, paths = self._beam_paths(context)
        masses = defaultdict(float)
        for decoded, mass in paths:
            if not decoded.startswith(prefix):
                continue
            rest = decoded[len(prefix):]
            end = re.search(r"[ .,]", rest)
            if end:
                word = prefix + rest[:end.start()]
                if word and re.fullmatch(r"[а-я]+", word):
                    masses[word] += mass
        total = sum(masses.values())
        if not total:
            return []
        return [(word, mass / total) for word, mass in sorted(masses.items(), key=lambda item: -item[1])[:n]]
