"""Fast model factory with explicit capability requirement.

Backends:
- ``ngram``: pretrained character n-gram file from models/ngram (default, see MA-10);
- ``kenlm``: word n-gram (KenLM, e.g. from a Hugging Face speech model "with LM"), turned
  into a next-letter distribution and mixed with the character n-gram (MA-36);
- ``mock_ngram``: character 5-gram trained at start-up on the small fallback corpus;
- ``hf_causal``: any causal language model from Hugging Face by its name (``Qwen/Qwen2.5-0.5B``)
  or a local folder; needs ``pip install -e ".[hf]"``.
"""
from eeg_speller.llm.backends.mock_ngram import MockNGram


def create_fast(cfg, corpus):
    backend = cfg["backend"]
    if backend == "ngram":
        from eeg_speller.llm.backends.char_ngram import CharNGram
        return CharNGram(cfg)
    if backend == "kenlm":
        from eeg_speller.llm.backends.kenlm_words import KenLMWords, default_char_model
        return KenLMWords(cfg, default_char_model())
    if backend == "mock_ngram":
        return MockNGram(cfg, corpus)
    if backend == "hf_causal":
        from eeg_speller.llm.backends.hf_causal import HFCausal
        return HFCausal(cfg)
    raise ValueError(f"fast backend {backend} does not provide full next-token distribution")
