from glob import translate
import re
import unicodedata


# TODO
"""
1. Normalize the corpus.
2. Pre-tokenize it into words.
3. Count how often each distinct pre-token appears. This gives a table like your words list.
4. Split each distinct word into characters and build the base vocabulary from those characters.
5. Run the merge-learning loop, weighting each pair by the word’s frequency.
"""


class Bpe:
    def __init__(self, languages: tuple[str, str] = ("en", "pt")):
        self.languages = languages
        self.original = languages[0]
        self.translated = languages[1]

    def normalization(text: str) -> str:
        unicode_normalized_seq = unicodedata.normalize("NFD", text)
        cleaned_unicode = re.sub(r"[\u0300-\u036f]", "", unicode_normalized_seq)
        return cleaned_unicode.lower().strip().replace("\n", " ").replace("\t", " ")

    def pre_tokenization(text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text)

    def single_char_split(pre_tokens: list[str]) -> list[str]:
        joined_words = "".join(pre_tokens)
        return [*joined_words]

    def compute_frequencies(self, pre_tokenized_list: list[str]) -> dict:
        words = {}
        for word in pre_tokenized_list:
            if word not in words:
                words[word] = 1
            else:
                words[word] += 1
        return words

    def compute_base_vocabulary():
        raise NotImplementedError

    def merge_pair():
        raise NotImplementedError

    def merge():
        raise NotImplementedError

    def tokenize():
        raise NotImplementedError
