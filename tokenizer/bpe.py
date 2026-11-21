import unicodedata
from collections import defaultdict

import regex as re

SPECIAL_SPACE_CHAR = "\u0120"
# NOTE
# use gpt-2 regex pattern to address (mostly) leading whitespace issue
GPT2_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class Bpe:
    """Byte Pair Enconding with byte-level tokenization.
    At first, BPE was implemented with character-level tokenization which would
    resulted in an incapacity to decode unkown IDs (without preserving the original char
    in some way). With byte-level, every Unicode char becomes UTF-8 bytes and every possible byte
    has a base token. This method is very effective for multi-lingual encoding and to address unknown
    characters not seen in training."""

    # TODO
    # I'll consider removing this languages param
    # bpe should only be responsible for tokenization, not handling the corpus

    def __init__(
        self, languages: tuple[str, str] = ("en", "pt"), max_tokens: int = 50257
    ):
        self.languages = languages
        self.original = languages[0]
        self.translated = languages[1]
        self.max_tokens = max_tokens
        self.vocabulary, self.encoding_lookup = self.bytes_to_unicode()
        self.words: dict[str, int] = defaultdict(int)
        self.splits: dict[str, list[str]] = None
        self.merges: dict[tuple[str, str], int] = {}

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
        return dict(zip(bs, cs)), dict(zip(cs, bs))

    def normalization(self, text: str) -> str:
        # NOTE
        # use NFC instead of NFD to preserve accentuated words
        # (specially in idioms that containes these chars (e.g pt-BR))
        # NFC maintains canonical equivalence (e.g both é and e + accent produce the same unicode code);
        # Does not alter compatibility chars (e.g ligatures);
        # ASCII and Latin-1 safe.

        # TODO
        # I'm considering not using this normalization, it could screw up
        # some decoding

        unicode_normalized_seq = unicodedata.normalize("NFC", text)
        cleaned_unicode = re.sub(r"[\u0300-\u036f]", "", unicode_normalized_seq)
        return cleaned_unicode

    def pre_tokenization(self, text: str) -> list[str]:
        return re.findall(GPT2_PATTERN, text)

    def tokenize(self, pre_tokenized_text: list[str]) -> list[list[str]]:
        return [
            [self.vocabulary[byte] for byte in word.encode("utf-8")]
            for word in pre_tokenized_text
        ]

    def compute_frequencies(self, pre_tokenized_list: list[str]) -> None:
        for word in pre_tokenized_list:
            self.words[word] += 1

    def build_splits(self) -> None:
        if self.words is not None:
            words = list(self.words)
            tokenized_words = self.tokenize(words)
            self.splits = dict(zip(words, tokenized_words))

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

            most_freq_pair = max(adj_pairs, key=lambda p: (adj_pairs[p], p))
            merge_rank = len(self.merges) + 1
            self.merges[most_freq_pair] = merge_rank
            self.vocabulary[256 + (merge_rank - 1)] = "".join(most_freq_pair)

            self.merge_pair(most_freq_pair)

    def train(self, corpus: list[str]):
        for text in corpus:
            normalized_text = self.normalization(text)
            pre_tokenized_text = self.pre_tokenization(normalized_text)
            self.compute_frequencies(pre_tokenized_text)

        self.build_splits()
        self.learn_merges()

    def encode(self, text: str) -> list[int]:
        normalized_text = self.normalization(text)
        pre_tokenized_text = self.pre_tokenization(normalized_text)
        tokenized_text = self.tokenize(pre_tokenized_text)
        return [self.encoding_lookup[c] for word in tokenized_text for c in word]

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8")
