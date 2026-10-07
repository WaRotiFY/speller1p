"""Local LLM correction through an OpenAI-compatible server on this machine.

Works with Ollama (http://127.0.0.1:11434/v1), llama.cpp `llama-server`
(http://127.0.0.1:8080/v1) and LM Studio (http://127.0.0.1:1234/v1).
Only loopback addresses are accepted, so participant text never leaves the computer.
"""
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse
import urllib.error
import urllib.request

LOOPBACK = {"127.0.0.1", "localhost", "::1"}
WORD = re.compile(r"[а-я]+")


def parse_words(text: str) -> list[str]:
    """Model reply -> lowercase Russian words in the speller alphabet."""
    text = text.strip().lower().replace("ё", "е")
    text = re.sub(r"^\s*ответ\s*:", "", text, flags=re.M)
    lines = [line for line in text.splitlines() if WORD.search(line)]
    # Small models sometimes add a preamble line; the answer is the last line with words.
    return WORD.findall(lines[-1]) if lines else []


class LocalOpenAICorrection:
    """Post-factum Text Correction with a local LLM; NEED-F-07; MA-17/20; Q-14, Q-33."""
    version = "local-openai-1"

    def __init__(self, cfg: dict):
        # IMPROV[Q-33]
        self.cfg = cfg
        self.base_url = cfg.get("base_url", "http://127.0.0.1:11434/v1").rstrip("/")
        host = urlparse(self.base_url).hostname
        if host not in LOOPBACK:
            raise ValueError(f"local_openai: base_url must point to this computer (127.0.0.1), got {host!r}")
        self.model_id = cfg["model_id"]
        self.timeout = float(cfg.get("timeout_s", 30))
        prompt_path = Path(cfg.get("prompt_file_local", "prompts/correction_local.txt"))
        if not prompt_path.is_absolute():
            prompt_path = Path(__file__).resolve().parents[3] / prompt_path
        self.prompt = prompt_path.read_text(encoding="utf-8")
        self.weights_hash = hashlib.sha256((self.model_id + "\n" + self.prompt).encode()).hexdigest()
        self._check_server()

    def _request(self, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.base_url + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _check_server(self) -> None:
        try:
            models = [m.get("id") for m in self._request("/models").get("data", [])]
        except (urllib.error.URLError, OSError) as exc:
            raise ValueError(f"local_openai: no LLM server at {self.base_url} "
                             f"(start Ollama or llama-server first): {exc}") from exc
        if models and self.model_id not in models:
            raise ValueError(f"local_openai: model {self.model_id!r} is not loaded; available: {models}")
        # Warm-up: Ollama loads weights on the first request, do it before the first word.
        self.correct("", ["тест"])

    def correct(self, left_context: str, window: list[str]) -> list[str]:
        # ASSUMPTION[MA-20]
        user = (f"Контекст: {left_context.strip() or '(начало текста)'}\n"
                f"Слова ({len(window)}): {' '.join(window)}")
        payload = {"model": self.model_id, "temperature": float(self.cfg.get("temperature", 0)),
                   "max_tokens": 16 * len(window) + 16, "stream": False,
                   "messages": [{"role": "system", "content": self.prompt},
                                {"role": "user", "content": user}]}
        reply = self._request("/chat/completions", payload)
        return parse_words(reply["choices"][0]["message"]["content"])
