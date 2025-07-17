import re
import time
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any

from autocomplete_system.models.base import AutocompleteModel


class NgramModel(AutocompleteModel):
    """Simple N-gram based autocompletion model."""

    def __init__(
            self,
            n=3,
    ):
        self.name = f"ngram_{n}"
        self.n = n
        self.word_freq = Counter()

        # Build n-gram dictionaries for each order (1 to n)
        self.ngrams = {k: defaultdict(Counter) for k in range(1, self.n + 1)}

        self.prefix_index: Dict[str, List[str]] = defaultdict(list)
        self.is_trained: bool = False
        self.training_time: float = 0.0
        self.vocab_size: int = 0

    def _tokenize(
            self,
            text: str,
    ) -> List[str]:
        return re.findall(r"\b[\w'-]{2,}\b", text.lower())

    def train(
            self,
            texts: List[str],
            min_freq: int = 2,
            verbose: bool = True,
            **_,
    ) -> None:
        start_time = time.time()

        # Process all texts and build n-grams
        for text in texts:
            # Tokenize the text into words
            words = self._tokenize(text)

            # Count word frequencies in the text
            for word in words:
                self.word_freq[word] += 1

            # Build n-grams for all orders from 1 to n
            for k in range(1, self.n + 1):
                for i in range(1, len(words) - k):
                    context = tuple(words[i:i + k])
                    next_word = words[i + k]
                    self.ngrams[k][context][next_word] += 1

        # Prune low-frequency words
        keep_words = Counter({word: freq for word, freq in self.word_freq.items() if freq >= min_freq})
        self.word_freq = self.word_freq & keep_words

        # Build prefix index for fast lookup
        for word in self.word_freq:
            for L in range(1, min(len(word), 10) + 1):
                prefix = word[:L]
                self.prefix_index[prefix].append(word)

        # Sort words in prefix index by frequency in descending order
        for p in self.prefix_index:
            self.prefix_index[p].sort(key=lambda w: self.word_freq[w], reverse=True)

        self.is_trained = True
        self.training_time = time.time() - start_time
        self.vocab_size = len(self.word_freq)

        if verbose:
            print(f"Training complete! Vocabulary size: {self.vocab_size}, Time: {self.training_time:.2f}s")

    def suggest(
            self,
            query: str,
            n_suggestions: int = 5,
    ) -> List[Tuple[str, float]]:
        if not self.is_trained:
            return []

        query = query.lower().strip()

        # Make sure query is not empty
        if not query:
            return []

        words = query.split()

        # If query ends with space, predict next word
        if query.endswith(' ') and words:
            for k in range(min(self.n, len(words)), 0, -1):
                context = tuple(words[-k:])
                if context in self.ngrams[k]:
                    candidates = self.ngrams[k][context].most_common(n_suggestions)
                    return [(word, self.word_freq[word]) for word, _ in candidates]
            # Either no context or no candidates found
            return []

        # Complete current word
        prefix = words[-1] if words else query

        if prefix in self.prefix_index:
            suggestions = self.prefix_index[prefix][:n_suggestions]
            return [(word, self.word_freq[word]) for word in suggestions]

        return []

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "training_time": self.training_time,
            "is_trained": self.is_trained,
            "n": self.n,
        }
