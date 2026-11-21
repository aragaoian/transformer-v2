import unicodedata
from collections import defaultdict
from itertools import count

import regex as re

SPECIAL_SPACE_CHAR = "\u0120"
# NOTE
# use gpt-2 regex pattern to address (mostly) leading whitespace issue
GPT2_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class Bpe:
    def __init__(
        self, languages: tuple[str, str] = ("en", "pt"), max_tokens: int = 50257
    ):
        self.languages = languages
        self.original = languages[0]
        self.translated = languages[1]
        self.max_tokens = max_tokens
        self.vocabulary: dict[int, str] = self.bytes_to_unicode()
        self.words: dict[str, int] = defaultdict(int)
        self.splits: dict[str, list[str]] = None
        self.merges: dict[tuple[str, str]] = defaultdict(count(1).__next__)

    def bytes_to_unicode(self):
        # NOTE
        # bs -> raw byte ints
        # cs -> unicode chars
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]

        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    def normalization(self, text: str) -> str:
        # NOTE
        # use NFC instead of NFD to preserve accentuated words
        # (specially in idioms that containes these chars (e.g pt-BR))
        # NFC maintains canonical equivalence (e.g both é and e + accent produce the same unicode code);
        # Does not alter compatibility chars (e.g ligatures);
        # ASCII and Latin-1 safe.
        unicode_normalized_seq = unicodedata.normalize("NFC", text)
        cleaned_unicode = re.sub(r"[\u0300-\u036f]", "", unicode_normalized_seq)
        return cleaned_unicode.strip()

    def pre_tokenization(self, text: str) -> list[str]:
        return [str(w).encode("utf-8") for w in re.findall(GPT2_PATTERN, text)]

    def compute_frequencies(self, pre_tokenized_list: list[str]) -> None:
        for word in pre_tokenized_list:
            self.words[word] += 1

    def build_splits(self) -> None:
        if self.words is not None:
            self.splits = {word: [*word] for word in self.words}

    def count_pairs(self):
        # TODO
        # optimize for larger corpus
        adj_pairs = defaultdict(int)
        for word, tokens in self.splits.items():
            for i in range(len(tokens) - 1):
                adj_pairs[(tokens[i], tokens[i + 1])] += self.words[word]
        return adj_pairs

    def merge_pair(self, pair: tuple[str, str]):
        # TODO
        # optimize for larger corpus
        merged = pair[0] + pair[1]
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

    def learn_merges(self):
        while len(self.vocabulary) < self.max_tokens:
            adj_pairs = self.count_pairs()
            if not adj_pairs:
                break

            # NOTE
            # .get depends on insertions order
            # use lambda
            most_freq_pair = max(adj_pairs, key=lambda p: (adj_pairs[p], p))
            merge_rank = self.merges[most_freq_pair]
            self.vocabulary[256 + merge_rank] = bytes(most_freq_pair).decode("utf-8")

            self.merge_pair(most_freq_pair)

    def train(self, corpus: list[str]):
        for phrase in corpus:
            normalized_phrase = self.normalization(phrase)
            pre_tokenized_phrase = self.pre_tokenization(normalized_phrase)
            self.compute_frequencies(pre_tokenized_phrase)

        self.build_splits()
        self.learn_merges()

    def encode(self) -> list[int]:
        raise NotImplementedError

    def decode(self):
        raise NotADirectoryError
