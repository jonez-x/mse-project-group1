import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from autocomplete_system.models.base import AutocompleteModel, AutocompleteResult


class NgramModel(AutocompleteModel):
    """
    Simple N-gram based autocompletion model.

    This model builds n-grams from the training texts and uses them
    to suggest completions based on the last words of the input query.

    Attributes:
        n (int): The maximum order of n-grams to use.
        word_freq (Counter): Frequency count of words in the training data.
        ngrams (Dict[int, Dict[Tuple[str], Counter]]): N-gram dictionaries for each order.
        prefix_index (Dict[str, List[str]]): Index for fast prefix-based suggestions.
        is_trained (bool): Flag indicating if the model has been trained.
        training_time (float): Time taken to train the model.
        vocab_size (int): Size of the vocabulary after training.
    """

    def __init__(
            self,
            n: int = 3,
    ) -> None:
        """
        Initialize the NgramModel with a specified n-gram order.

        Args:
            n (int): The maximum order of n-grams to use (default: 3).
        """
        # Initialize the base class with a name and set up the model
        super().__init__(name=f"ngram_{n}")
        self.n = n

        # Frequency count of words in the training data
        self.word_freq = Counter()

        # Build n-gram dictionaries for each order (1 to n)
        self.ngrams = {k: defaultdict(Counter) for k in range(1, self.n + 1)}
        self.vocab_size: int = 0

        # Dictionary for fast prefix-based suggestions
        self.prefix_index: Dict[str, List[str]] = defaultdict(list)

        # Initialize training status and timing
        self.is_trained: bool = False
        self.training_time: float = 0.0

    def _tokenize(
            self,
            text: str,
    ) -> List[str]:
        """
        Helper method to tokenize text into words.

        Args:
            text (str): Input text to tokenize.

        Returns:
            List[str]: List of words extracted from the text.
        """
        return re.findall(r"\b[\w'-]{2,}\b", text.lower())

    def train(
            self,
            texts: List[str],
            min_freq: int = 2,
            verbose: bool = True,
            **_,
    ) -> None:
        """
        Train the N-gram model on a list of texts.

        This method processes the texts, builds n-grams, and creates a prefix index for fast lookups.

        Args:
            texts (List[str]): List of input texts to train on.
            min_freq (int): Minimum frequency for words to be included in the vocabulary (default: 2).
            verbose (bool): Whether to print training progress (default: True).
            **_: Additional keyword arguments (not used).
        """
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
                for i in range(len(words) - k):
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
    ) -> List[AutocompleteResult]:
        # If not trained, return empty suggestions --> Need to handle this in the frontend
        if not self.is_trained:
            return []

        # Use the parent class logic
        return super().suggest(query, n_suggestions)

    def _predict_next_word(
            self,
            words: List[str],
            n_suggestions: int = 5,
    ) -> List[AutocompleteResult]:
        """
        Predict the next word using n-gram analysis.

        Args:
            words (List[str]): List of preceding words.
            n_suggestions (int): Maximum number of suggestions to return.

        Returns:
            List[AutocompleteResult]: List of suggested next words.
        """
        # Try different n-gram orders, starting from highest
        for k in range(min(self.n, len(words)), 0, -1):
            context = tuple(words[-k:])
            if context in self.ngrams[k]:
                candidates = self.ngrams[k][context].most_common(n_suggestions)
                if candidates:  # Only return if we actually have candidates
                    return [
                        AutocompleteResult(
                            word=word,
                            score=float(freq),
                            type="next_word",
                        ) for word, freq in candidates
                    ]

        # No n-gram context found
        return []

    def _complete_word(
            self,
            partial_word: str,
            n_suggestions: int = 5,
    ) -> List[AutocompleteResult]:
        """
        Complete a partial word using the prefix index.

        Args:
            partial_word (str): The partial word to complete.
            n_suggestions (int): Maximum number of suggestions to return.

        Returns:
            List[AutocompleteResult]: List of word completions.
        """
        if partial_word in self.prefix_index:
            suggestions = self.prefix_index[partial_word][:n_suggestions]
            return [
                AutocompleteResult(
                    word=word,
                    score=float(self.word_freq[word]),
                    type="completion",
                ) for word in suggestions
            ]

        return []

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "training_time": self.training_time,
            "is_trained": self.is_trained,
            "n": self.n,
        }

    def is_training_required(self) -> bool:
        return True
