import re
import unicodedata


class Bpe:
    def __init__(self):
        pass

    def normalization(text: str) -> str:
        unicode_normalized_seq = unicodedata.normalize("NFD", text)
        cleaned_unicode = re.sub(r"[\u0300-\u036f]", "", unicode_normalized_seq)
        return cleaned_unicode.lower().strip().replace("\n", " ").replace("\t", " ")

    def pre_tokenization(text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text)

    def single_char_split(pre_tokens: list[str]) -> list[str]:
        joined_words = "".join(pre_tokens)
        return [*joined_words]

    def merge():
        raise NotImplementedError
