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
        self.vocabulary = [chr(i) for i in range(128)]
        self.words = {}

    def normalization(self, text: str) -> str:
        unicode_normalized_seq = unicodedata.normalize("NFD", text)
        cleaned_unicode = re.sub(r"[\u0300-\u036f]", "", unicode_normalized_seq)
        return cleaned_unicode.lower().strip().replace("\n", " ").replace("\t", " ")

    def pre_tokenization(self, text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text)

    def single_char_split(self, pre_tokens: list[str]) -> list[str]:
        joined_words = "".join(pre_tokens)
        return [*joined_words]

    def compute_frequencies(self, pre_tokenized_list: list[str]) -> dict:
        for word in pre_tokenized_list:
            if word not in self.words:
                self.words[word] = 1
            else:
                self.words[word] += 1

    def merge(self, pre_tokens: list[str], splitted_pre_tokens: list[str]):
        """1. Take the rules in the order they were learned, from first to last.
        2. For the current rule, scan the word’s symbols from left to right, looking for places where the two symbols of the rule sit side by side.
        3. Wherever the pair is found, replace the two symbols with the single merged symbol.
        4. Move to the next rule and repeat on the updated sequence.
        5. When all rules have been tried, whatever symbols remain are the final tokens for that word.
        """

        adj_pairs = {}
        for i in range(len(splitted_pre_tokens) - 1):
            adj_pair = splitted_pre_tokens[i] + splitted_pre_tokens[i + 1]
            for pre_token in pre_tokens:
                if adj_pair in pre_token:
                    if adj_pair not in adj_pairs:
                        adj_pairs[adj_pair] = 1
                    else:
                        adj_pairs[adj_pair] += 1

        rules = [max(adj_pairs, key=adj_pairs.get)]
        for rule in rules:
            for i in range(len(splitted_pre_tokens) - 1):
                adj_pair = splitted_pre_tokens[i] + splitted_pre_tokens[i + 1]
                for pre_token in pre_tokens:
                    if adj_pair in pre_token:
                        rules.append(adj_pair)
                    else:
                        rules.extend(
                            [splitted_pre_tokens[i], splitted_pre_tokens[i + 1]]
                        )

        print(rules)

    def tokenize(self, corpus: list[str]):
        for phrase in corpus:
            normalized_phrase = self.normalization(phrase)
            pre_tokenized_phrase = self.pre_tokenization(normalized_phrase)
            self.compute_frequencies(pre_tokenized_phrase)
            splitted_pre_tokens = self.single_char_split(pre_tokenized_phrase)
            self.merge(pre_tokenized_phrase, splitted_pre_tokens)
