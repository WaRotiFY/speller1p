"""Word n-gram (KenLM) as the fast model: next-letter distribution from a word language model.

Any KenLM model works: ``.arpa``, ``.arpa.gz`` or a KenLM binary, with an optional
``unigrams.txt`` word list next to it (the layout of Hugging Face speech models
"with LM"). ``fast_llm.model_id`` is either a local file or folder, or a Hugging Face
repository name such as ``jonatasgrosman/wav2vec2-large-xlsr-53-russian``; the files
are downloaded once into ``models/kenlm/<repository>/`` (``allow_download``).

How a word model gives the next letter (MA-36). Let ``u`` be the unfinished word and
``ctx`` the finished words of the current sentence. For every vocabulary word ``w``
that starts with ``u``:

    P(next = c | ctx, u)  ∝  Σ P(w | ctx) over words w = u·c·…        (letters)
    P(next = boundary)    ∝  P(w = u | ctx)                           (u is a whole word)

The boundary mass is split into ``.`` (the model's P(</s> | ctx, u)), ``,`` (fixed
share) and space. If no vocabulary word starts with ``u`` (a typo already typed, a rare
word) the word part is empty and the character n-gram takes over: the result is always
mixed with the character 4-gram, ``(1 − λ)·word + λ·char``.
"""
from __future__ import annotations

import bisect
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import time
import urllib.request

import numpy as np

from eeg_speller.core.distributions import SymbolDistribution, log_probs
from eeg_speller.core.keyboard import LM_ALPHABET, keyboard_text, project

PROJECT = Path(__file__).resolve().parents[3]
CACHE = PROJECT / "models" / "kenlm"
WORD = re.compile(r"[а-яё]+")
DEFAULT_REPO = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
MODEL_SUFFIXES = (".binary", ".bin", ".klm", ".arpa", ".arpa.gz")

# Progress of a running download, shown in the operator window.
download_status: dict = {}


def _is_model_file(name: str) -> bool:
    return name.endswith(MODEL_SUFFIXES)


def _download(url: str, dest: Path, label: str) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    start = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "eeg-speller/0.2",
                                               **({"Range": f"bytes={start}-"} if start else {})})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0) + start
        if start and resp.status != 206:          # server ignored the range: start again
            start = 0
        with open(tmp, "ab" if start else "wb") as fh:
            done = start
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                download_status.update(file=label, done_mb=round(done / 1e6), total_mb=round(total / 1e6))
    tmp.replace(dest)


def fetch_hf_repo(repo: str, allow_download: bool) -> Path:
    """Folder in models/kenlm with the KenLM file (+ unigrams.txt) of a Hugging Face repo."""
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("Hugging Face name must look like owner/name")
    folder = CACHE / repo.replace("/", "__")
    manifest = folder / "files.json"
    if manifest.is_file():
        return folder
    if not allow_download:
        raise ValueError(f"{repo} is not downloaded yet: tick 'allow download' (about 1.5 GB)")
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=30) as resp:
        info = json.load(resp)
    names = [s["rfilename"] for s in info.get("siblings", [])]
    models = [n for n in names if _is_model_file(n)]
    if not models:
        raise ValueError(f"{repo} has no KenLM file ({', '.join(MODEL_SUFFIXES)})")
    model = sorted(models, key=lambda n: (not n.startswith("language_model/"), n))[0]
    wanted = [model] + [n for n in names if n.endswith("unigrams.txt") and Path(n).parent == Path(model).parent]
    folder.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        dest = folder / Path(name).name
        if not dest.exists():
            _download(f"https://huggingface.co/{repo}/resolve/main/{name}", dest, Path(name).name)
    manifest.write_text(json.dumps({"repo": repo, "files": wanted, "license": (info.get("cardData") or {}).get("license"),
                                    "sha": info.get("sha"), "downloaded": time.strftime("%Y-%m-%d")},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    download_status.clear()
    return folder


def resolve_model(model_id: str, allow_download: bool) -> tuple[Path, Path | None, dict]:
    """model_id -> (KenLM file, unigrams.txt or None, provenance)."""
    path = Path(model_id).expanduser()
    if path.exists():
        meta = {"source": "local", "path": str(path.resolve())}
    else:
        path = fetch_hf_repo(model_id, allow_download)
        meta = json.loads((path / "files.json").read_text(encoding="utf-8"))
        meta["source"] = "huggingface"
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if _is_model_file(p.name))
        if not files:
            raise ValueError(f"no KenLM file in {path}")
        model_file = files[0]
    else:
        model_file = path
    unigrams = model_file.parent / "unigrams.txt"
    return model_file, unigrams if unigrams.is_file() else None, meta


def read_vocabulary(model_file: Path, unigrams: Path | None) -> list[str]:
    if unigrams is not None:
        return [w.strip() for w in unigrams.read_text(encoding="utf-8", errors="ignore").splitlines() if w.strip()]
    if model_file.name.endswith((".arpa", ".arpa.gz")):
        opener = gzip.open if model_file.name.endswith(".gz") else open
        words, inside = [], False
        with opener(model_file, "rt", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.startswith("\\1-grams:"):
                    inside = True
                    continue
                if inside:
                    if not line.strip() or line.startswith("\\"):
                        break
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        words.append(parts[1].strip())
        return words
    raise ValueError("a binary KenLM model needs unigrams.txt next to it (the word list)")


class KenLMWords:
    """Source Model (Fast LLM): word KenLM -> next letter; NEED-F-06/08; MA-08/11/36; Q-13, Q-47."""
    version = "kenlm-words@1"

    def __init__(self, cfg: dict, char_model=None):
        # IMPROV[Q-47]
        try:
            import kenlm
        except ImportError as exc:
            raise ValueError('не установлен пакет kenlm: uv pip install -e ".[kenlm]" '
                             "(сборка на Windows и Python 3.13 — models/kenlm/README.md)") from exc
        self.kenlm = kenlm
        self.cfg = cfg
        model_id = str(cfg.get("model_id") or DEFAULT_REPO).strip()
        self.model_file, unigrams, self.provenance = resolve_model(model_id, bool(cfg.get("allow_download")))
        self.model = kenlm.Model(str(self.model_file))
        self.order = self.model.order
        self.model_id = model_id
        st = self.model_file.stat()
        self.weights_hash = hashlib.sha256(f"{self.model_file.name}|{st.st_size}|{int(st.st_mtime)}|"
                                           f"{self.provenance.get('sha', '')}".encode()).hexdigest()
        self.char_mix = float(cfg.get("char_mix", 0.1))
        self.comma_share = float(cfg.get("comma_share", 0.05))
        self.char_model = char_model
        # Vocabulary: Russian words only, the most probable ones (unigram score) first.
        limit = int(cfg.get("vocab_limit", 60000))
        raw = [w for w in read_vocabulary(self.model_file, unigrams) if WORD.fullmatch(w.lower())]
        scored = []
        for w in raw:
            scored.append((self.model.score(w, bos=False, eos=False), w))
        scored.sort(reverse=True)
        self.original: dict[str, str] = {}     # normalized (е) -> spelling known to the model
        for _, w in scored:
            key = w.lower().replace("ё", "е")
            self.original.setdefault(key, w)
            if len(self.original) >= limit:
                break
        self.words = sorted(self.original)     # sorted for prefix search
        self.vocab_size = len(raw)

    # ------------------------------------------------------------------ helpers
    def _state(self, finished: list[str], sentence_start: bool):
        state, out = self.kenlm.State(), self.kenlm.State()
        if sentence_start:
            self.model.BeginSentenceWrite(state)
        else:
            self.model.NullContextWrite(state)
        for w in finished[-(self.order - 1):]:
            self.model.BaseScore(state, self.original.get(w, w), out)
            state, out = out, state
        return state

    def _parse(self, context: str):
        """Typed text -> (finished words of the current sentence, unfinished word, sentence start).

        The whole text is typed from its beginning, so the current sentence starts with <s>.
        """
        text = keyboard_text(context)
        sentence = text.split(".")[-1]
        prefix = re.search(r"[а-я]*$", sentence).group()
        finished = re.findall(r"[а-я]+", sentence[:len(sentence) - len(prefix)])
        return finished, prefix, True

    def _candidates(self, prefix: str):
        lo = bisect.bisect_left(self.words, prefix)
        hi = bisect.bisect_left(self.words, prefix + "￿")
        return self.words[lo:hi]

    def _word_scores(self, finished, prefix, sentence_start):
        state = self._state(finished, sentence_start)
        out = self.kenlm.State()
        cands = self._candidates(prefix)
        scores = np.fromiter((self.model.BaseScore(state, self.original[w], out) for w in cands),
                             dtype=float, count=len(cands))
        return cands, scores * math.log(10), state

    # --------------------------------------------------------------- interface
    def lm_distribution(self, context: str) -> np.ndarray:
        finished, prefix, sentence_start = self._parse(context)
        cands, logp, state = self._word_scores(finished, prefix, sentence_start)
        word = np.zeros(len(LM_ALPHABET))
        if len(cands):
            p = np.exp(logp - logp.max())
            boundary = 0.0
            n = len(prefix)
            idx = {c: i for i, c in enumerate(LM_ALPHABET)}
            for w, pw in zip(cands, p):
                if len(w) > n:
                    word[idx[w[n]]] += pw
                else:
                    boundary += pw
            if boundary > 0 and prefix:
                # Period = end of sentence after this word; comma = fixed share; the rest is a space.
                out = self.kenlm.State()
                after = self.kenlm.State()
                self.model.BaseScore(state, self.original.get(prefix, prefix), after)
                p_end = 10 ** self.model.BaseScore(after, "</s>", out)
                word[idx["."]] += boundary * p_end
                word[idx[","]] += boundary * (1 - p_end) * self.comma_share
                word[idx[" "]] += boundary * (1 - p_end) * (1 - self.comma_share)
            if word.sum() > 0:
                word /= word.sum()
        char = self.char_model.lm_distribution(context) if self.char_model is not None else \
            np.full(len(LM_ALPHABET), 1 / len(LM_ALPHABET))
        if word.sum() == 0:
            return char
        mixed = (1 - self.char_mix) * word + self.char_mix * char
        return mixed / mixed.sum()

    def predict(self, context, repertoire):
        # ASSUMPTION[MA-36]
        cm = self.cfg.get("char_marginal", {})
        p = project(self.lm_distribution(context), repertoire.symbols,
                    float(cm.get("backspace_prob", 0.02)), float(self.cfg.get("other_prob", 0.005)))
        return SymbolDistribution(repertoire, log_probs(p), f"fast_llm:{self.model_id}", len(context))

    def top_words(self, context: str, n: int):
        # ASSUMPTION[MA-11]
        finished, prefix, sentence_start = self._parse(context)
        cands, logp, _ = self._word_scores(finished, prefix, sentence_start)
        mask = np.array([len(w) > len(prefix) for w in cands], dtype=bool) if len(cands) else np.zeros(0, bool)
        if not mask.any():
            return []
        cands = [w for w, m in zip(cands, mask) if m]
        logp = logp[mask]
        order = np.argsort(-logp)[:n]
        p = np.exp(logp[order] - logp[order].max())
        p /= p.sum()
        return [(cands[i], float(q)) for i, q in zip(order, p)]


def downloaded_models() -> list[dict]:
    out = []
    if CACHE.is_dir():
        for f in sorted(CACHE.glob("*/files.json")):
            try:
                meta = json.loads(f.read_text(encoding="utf-8"))
                out.append({"id": meta["repo"], "license": meta.get("license")})
            except (OSError, ValueError, KeyError):
                continue
    return out


def kenlm_available() -> bool:
    try:
        import kenlm  # noqa: F401
        return True
    except ImportError:
        return False


def default_char_model():
    from eeg_speller.llm.backends.char_ngram import CharNGram, DEFAULT_MODEL
    return CharNGram({"model_file": DEFAULT_MODEL.name}) if DEFAULT_MODEL.is_file() else None

