import re
import unicodedata
from collections import defaultdict


# TODO
# 1. See more about gpt-2 space handle style
# 2. test on the corpus

SPECIAL_SPACE_CHAR = "\u0120"


class Bpe:
    def __init__(
        self, languages: tuple[str, str] = ("en", "pt"), max_tokens: int = 50000
    ):
        self.languages = languages
        self.original = languages[0]
        self.translated = languages[1]
        self.max_tokens = max_tokens
        self.vocabulary = [chr(i) for i in range(128)]
        self.words: dict[str, int] = defaultdict(int)
        self.splits: dict[str, list[str]] = None
        self.merges: list[tuple[str, str]] = []

    def normalization(self, text: str) -> str:
        unicode_normalized_seq = unicodedata.normalize("NFD", text)
        cleaned_unicode = re.sub(r"[\u0300-\u036f]", "", unicode_normalized_seq)
        return cleaned_unicode.lower().strip().replace("\n", " ").replace("\t", " ")

    def pre_tokenization(self, text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text)

    def compute_frequencies(self, pre_tokenized_list: list[str]) -> dict:
        for word in pre_tokenized_list:
            self.words[word] += 1

    def build_splits(self) -> None:
        if self.words is not None:
            self.splits = {word: [*word] for word in self.words}

    def count_pairs(self):
        adj_pairs = defaultdict(int)
        for word, tokens in self.splits.items():
            for i in range(len(tokens) - 1):
                adj_pairs[(tokens[i], tokens[i + 1])] += self.words[word]
        return adj_pairs

    def merge_pair(self, pair: tuple[str, str]):
        merged = "".join(pair)
        for word, tokens in self.splits.items():
            new_split = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_split.append(merged)
                    i += 2
                else:
                    new_split.append(tokens[i])
                    i += 1
            self.splits[word] = new_split

    def tokenize(self):
        while len(self.vocabulary) < self.max_tokens:
            adj_pairs = self.count_pairs()
            print(adj_pairs)
            if not adj_pairs:
                break

            most_freq_pair = max(adj_pairs, key=adj_pairs.get)
            self.merges.append(most_freq_pair)
            self.vocabulary.append("".join(most_freq_pair))

            self.merge_pair(most_freq_pair)

    def train(self, corpus: list[str]):
        for phrase in corpus:
            normalized_phrase = self.normalization(phrase)
            pre_tokenized_phrase = self.pre_tokenization(normalized_phrase)
            self.compute_frequencies(pre_tokenized_phrase)

        self.build_splits()
        self.tokenize()
