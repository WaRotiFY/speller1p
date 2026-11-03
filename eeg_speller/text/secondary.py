"""Version of text after accepted corrections."""


class SecondaryText:
    """Secondary (Corrected) Text; NEED-F-07; MA-19/20; Q-16."""
    def __init__(self):
        self.value = ""
        self.frozen_upto_word = 0
        self.source_primary = ""

    def synchronize(self, primary: str) -> None:
        if primary.startswith(self.source_primary):
            self.value += primary[len(self.source_primary):]
        elif self.source_primary.startswith(primary):
            self.value = self.value[:-(len(self.source_primary) - len(primary))]
        else:
            raise ValueError("primary text changed outside append/delete contract")
        self.source_primary = primary

    def replace_window(self, words_out: list[str], start_word: int) -> None:
        import re
        spans = list(re.finditer(r"[а-яa-z]+", self.value))
        if not words_out or start_word >= len(spans):
            return
        selected = spans[start_word:start_word + len(words_out)]
        if len(selected) != len(words_out):
            raise ValueError("correction window does not match text")
        for match, word in reversed(list(zip(selected, words_out))):
            self.value = self.value[:match.start()] + word + self.value[match.end():]
        self.frozen_upto_word = max(self.frozen_upto_word, start_word)
