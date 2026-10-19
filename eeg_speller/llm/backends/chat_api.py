"""Slow model (post-factum correction) through any OpenAI-compatible chat API.

Providers are presets of a base URL; the model is chosen from the provider's own list.
Free options as of October 2026 (they need a free key, no card; limits change often):
OpenRouter models ending in ``:free``, Groq, NVIDIA NIM. ``ollama`` runs on this
computer. Keys are stored only on this computer (``~/.eeg_speller/keys.json``) or in
an environment variable, and never in session recordings.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

PROJECT = Path(__file__).resolve().parents[3]
PROMPTS = PROJECT / "prompts"
DEFAULT_PROMPT = PROMPTS / "correction_v2.txt"
USER_PROMPT = PROMPTS / "user" / "correction.txt"
KEYS_FILE = Path(os.environ.get("EEG_SPELLER_HOME", Path.home() / ".eeg_speller")) / "keys.json"
SPLIT = "---- сообщение ----"
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
WORD = re.compile(r"[а-я]+")

PROVIDERS = {
    "ollama": {"label": "Ollama на этом компьютере", "base_url": "http://127.0.0.1:11434/v1",
               "remote": False, "key_env": None, "suggested": "qwen2.5:3b",
               "help": "ollama pull qwen2.5:3b — затем модель появится в списке"},
    "openrouter": {"label": "OpenRouter — бесплатные модели (:free)", "base_url": "https://openrouter.ai/api/v1",
                   "remote": True, "key_env": "OPENROUTER_API_KEY", "free_suffix": ":free",
                   "suggested": "meta-llama/llama-3.3-70b-instruct:free",
                   "help": "Бесплатный ключ без карты: openrouter.ai/keys. Лимит ~50 запросов в день."},
    "groq": {"label": "Groq — бесплатный ключ", "base_url": "https://api.groq.com/openai/v1",
             "remote": True, "key_env": "GROQ_API_KEY", "suggested": "openai/gpt-oss-120b",
             "help": "Ключ: console.groq.com/keys. ~30 запросов в минуту, ~1000 в день."},
    "nvidia": {"label": "NVIDIA NIM — бесплатный ключ", "base_url": "https://integrate.api.nvidia.com/v1",
               "remote": True, "key_env": "NVIDIA_API_KEY", "suggested": "meta/llama-3.3-70b-instruct",
               "help": "Ключ: build.nvidia.com (нужна регистрация в NVIDIA Developer)."},
    "custom": {"label": "Свой адрес (OpenAI-совместимый)", "base_url": "",
               "remote": None, "key_env": "SPELLER_LLM_API_KEY", "suggested": "",
               "help": "Например llama.cpp server http://127.0.0.1:8080/v1 или LM Studio."},
}


# ---------------------------------------------------------------------------- keys
def _read_keys() -> dict:
    try:
        return json.loads(KEYS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_key(provider: str, key: str) -> None:
    if provider not in PROVIDERS:
        raise ValueError("unknown provider")
    keys = _read_keys()
    if key:
        keys[provider] = key.strip()
    else:
        keys.pop(provider, None)
    KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    KEYS_FILE.write_text(json.dumps(keys, indent=2), encoding="utf-8")
    try:
        os.chmod(KEYS_FILE, 0o600)
    except OSError:
        pass


def get_key(provider: str) -> str | None:
    env = PROVIDERS.get(provider, {}).get("key_env")
    return (os.environ.get(env) if env else None) or _read_keys().get(provider)


def provider_list() -> list[dict]:
    return [{"id": pid, "label": p["label"], "base_url": p["base_url"], "remote": p["remote"],
             "needs_key": p["key_env"] is not None, "has_key": bool(get_key(pid)),
             "suggested": p["suggested"], "help": p["help"]} for pid, p in PROVIDERS.items()]


def base_url_for(provider: str, base_url: str | None) -> str:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}")
    url = (base_url or PROVIDERS[provider]["base_url"]).strip().rstrip("/")
    if not url:
        raise ValueError("enter the base URL of the API (…/v1)")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("base URL must start with http:// or https://")
    if parsed.scheme == "http" and parsed.hostname not in LOOPBACK:
        raise ValueError("plain http is allowed only for this computer; use https")
    return url


def is_remote(url: str) -> bool:
    return urlparse(url).hostname not in LOOPBACK


def _request(url: str, key: str | None, payload: dict | None = None, timeout: float = 30.0) -> dict:
    headers = {"Content-Type": "application/json", "User-Agent": "eeg-speller/0.2"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if "openrouter.ai" in url:
        headers["HTTP-Referer"] = "http://127.0.0.1"
        headers["X-Title"] = "EEG speller"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        hint = {401: " — неверный ключ", 403: " — доступ запрещён: возможно, провайдер недоступен из вашей страны",
                429: " — исчерпан лимит запросов"}.get(exc.code, "")
        raise ValueError(f"HTTP {exc.code} from {urlparse(url).hostname}{hint}: {body}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise ValueError(f"no connection to {urlparse(url).hostname}: {exc}") from exc


def list_models(provider: str, base_url: str | None = None, only_free: bool = True) -> list[str]:
    url = base_url_for(provider, base_url)
    data = _request(url + "/models", get_key(provider), timeout=20)
    items = data.get("data", data if isinstance(data, list) else [])
    ids = []
    for m in items:
        mid = m.get("id") if isinstance(m, dict) else None
        if not mid:
            continue
        suffix = PROVIDERS[provider].get("free_suffix")
        if only_free and suffix:
            pricing = m.get("pricing") or {}
            free = mid.endswith(suffix) or (str(pricing.get("prompt")) in ("0", "0.0")
                                             and str(pricing.get("completion")) in ("0", "0.0"))
            if not free:
                continue
        ids.append(mid)
    return sorted(set(ids))


# ------------------------------------------------------------------------- prompt
def read_prompt(which: str = "current") -> str:
    if which == "current" and USER_PROMPT.is_file():
        return USER_PROMPT.read_text(encoding="utf-8")
    return DEFAULT_PROMPT.read_text(encoding="utf-8")


def save_prompt(text: str) -> None:
    check_template(text)
    USER_PROMPT.parent.mkdir(parents=True, exist_ok=True)
    USER_PROMPT.write_text(text, encoding="utf-8")


def reset_prompt() -> None:
    if USER_PROMPT.is_file():
        USER_PROMPT.unlink()


def split_template(text: str) -> tuple[str, str]:
    if SPLIT not in text:
        raise ValueError(f"prompt must contain the line '{SPLIT}' between system part and message")
    system, user = text.split(SPLIT, 1)
    return system.strip(), user.strip("\n")


def check_template(text: str) -> None:
    _, user = split_template(text)
    for field in ("{context}", "{words}"):
        if field not in user:
            raise ValueError(f"message template must contain {field}")
    try:
        user.format(context="", n=0, words="", alternatives="")
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"message template: unknown placeholder or stray brace ({exc}); "
                         "allowed: {context} {n} {words} {alternatives}") from exc


def format_alternatives(words: list[str], letters: list[list[list]] | None) -> str:
    """letters[i][j] = [[symbol, p], ...] for character j of word i (top-K of the posterior)."""
    if not letters:
        return ""
    lines = ["Варианты букв:"]
    for word, chars in zip(words, letters):
        parts = []
        for ch, alts in zip(word, chars):
            if not alts:
                parts.append(f"{ch}[?]")
                continue
            parts.append(f"{ch}[" + " ".join(f"{s if s != ' ' else '␣'} {round(p * 100)}" for s, p in alts) + "]")
        lines.append(f"{word}: " + " ".join(parts))
    return "\n".join(lines)


def build_messages(template: str, left_context: str, window: list[str], letters=None) -> list[dict]:
    system, user = split_template(template)
    text = user.format(context=left_context.strip() or "(начало текста)", n=len(window),
                       words=" ".join(window), alternatives=format_alternatives(window, letters))
    return [{"role": "system", "content": system}, {"role": "user", "content": text.strip()}]


def parse_words(text: str) -> list[str]:
    """Model reply -> lowercase Russian words in the speller alphabet."""
    text = text.strip().lower().replace("ё", "е")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^\s*ответ\s*:", "", text, flags=re.M)
    lines = [line for line in text.splitlines() if WORD.search(line)]
    return WORD.findall(lines[-1]) if lines else []


class ChatCorrection:
    """Post-factum Text Correction through a chat API; NEED-F-07; MA-17/20; Q-14, Q-33, Q-41."""
    version = "chat-api@1"

    def __init__(self, cfg: dict, allow_remote: bool):
        # IMPROV[Q-33]
        # IMPROV[Q-41]
        self.cfg = cfg
        self.provider = cfg.get("provider", "ollama")
        self.base_url = base_url_for(self.provider, cfg.get("base_url_override") or
                                     (cfg.get("base_url") if self.provider == "custom" else None))
        self.remote = is_remote(self.base_url)
        if self.remote and not allow_remote:
            raise PermissionError("cloud correction sends the typed text to a third party: "
                                  "not allowed by this profile (llm.allow_remote: false)")
        self.model_id = str(cfg.get("model_id") or "").strip()
        if not self.model_id:
            raise ValueError("choose a model for the slow LLM")
        self.key = get_key(self.provider)
        if PROVIDERS[self.provider]["key_env"] and self.remote and not self.key:
            raise ValueError(f"no API key for {self.provider}: paste it in the operator window")
        self.timeout = float(cfg.get("timeout_s", 30))
        self.template = cfg.get("prompt_text") or read_prompt()
        check_template(self.template)
        self.prompt_hash = hashlib.sha256(self.template.encode()).hexdigest()
        self.weights_hash = hashlib.sha256(f"{self.provider}|{self.base_url}|{self.model_id}|"
                                           f"{self.template}".encode()).hexdigest()
        self.last_messages: list[dict] | None = None
        self.last_reply: str | None = None

    def correct(self, left_context: str, window: list[str], letters=None) -> list[str]:
        # ASSUMPTION[MA-20]
        messages = build_messages(self.template, left_context, window, letters)
        payload = {"model": self.model_id, "temperature": float(self.cfg.get("temperature", 0)),
                   # Reasoning models (gpt-oss, deepseek-r1, qwen3…) spend tokens on thinking before the answer.
                   "max_tokens": int(self.cfg.get("max_tokens") or max(1024, 24 * len(window) + 32)), "stream": False,
                   "messages": messages}
        self.last_messages = messages
        for attempt in range(2):
            try:
                reply = _request(self.base_url + "/chat/completions", self.key, payload, self.timeout)
                break
            except ValueError as exc:
                # Free tiers answer 429 when busy: one short retry, then give up for this window.
                if attempt or "HTTP 429" not in str(exc):
                    raise
                time.sleep(2.0)
        choice = reply["choices"][0]
        content = choice["message"].get("content") or ""
        self.last_reply = content
        if not content.strip():
            why = choice.get("finish_reason")
            raise ValueError("empty reply" + (" (finish_reason=length: the model used the whole token limit, "
                                              "probably on reasoning; raise slow_llm.max_tokens or pick another model)"
                                              if why == "length" else f" (finish_reason={why})"))
        return parse_words(content)
