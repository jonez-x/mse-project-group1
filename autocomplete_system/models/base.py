from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class AutocompleteResult:
    """
    Result object for autocomplete suggestions.

    Attributes:
        word (str): The suggested word.
        score (Optional[float]): The confidence score of the suggestion, if applicable.
        type (str): The type of suggestion, e.g., "completion" or "next_word" (default: "completion").
    """
    word: str
    score: Optional[float] = None
    type: str = "completion"

    def __str__(self):
        return f"AutocompleteResult(word={self.word}, score={self.score}, type={self.type})"


class AutocompleteModel(ABC):
    """
    Abstract base class for autocompletion models.

    Supports both training-based models (like N-gram) and
    pretrained/API-based models (like DataMuse).
    """
    name: str

    def __init__(self, name: str) -> None:
        """
        Initialize the model with a name.

        Args:
            name (str): The name of the model.
        """
        self.name = name

    @abstractmethod
    def train(
            self,
            texts: List[str],
            **kwargs,
    ) -> None:
        """
        Build internal data structures.

        For API-based models, this method may be a no-op.
        For training-based models, this should process the texts.
        """
        pass

    def suggest(
            self,
            query: str,
            n_suggestions: int = 3,
    ) -> List[AutocompleteResult]:
        """
        Get autocomplete suggestions for the given query.

        Args:
            query (str): Input text to get suggestions for.
            n_suggestions (int): Maximum number of suggestions to return (default: 3).

        Returns:
            List[AutocompleteResult]: List of autocomplete results with words and scores.
        """
        # For empty queries, return an empty list
        if not query:
            return []

        # Determine if we should complete a word or predict next word
        if query.endswith(' '):
            # Predict next word
            words = query.lower().strip().split()
            if words:
                return self._predict_next_word(words, n_suggestions)
        else:
            # Complete current word
            words = query.lower().strip().split()
            if words:
                current_word = words[-1]
                if len(current_word) >= 1:  # Allow single character prefixes
                    return self._complete_word(current_word, n_suggestions)

        return []

    @abstractmethod
    def _predict_next_word(
            self,
            words: List[str],
            n_suggestions: int = 3,
    ) -> List[AutocompleteResult]:
        """
        Predict the next word given a list of preceding words.

        Args:
            words (List[str]): List of preceding words.
            n_suggestions (int): Maximum number of suggestions to return.

        Returns:
            List[AutocompleteResult]: List of suggested next words.
        """
        pass

    @abstractmethod
    def _complete_word(
            self,
            partial_word: str,
            n_suggestions: int = 3,
    ) -> List[AutocompleteResult]:
        """
        Complete a partial word.

        Args:
            partial_word (str): The partial word to complete.
            n_suggestions (int): Maximum number of suggestions to return.

        Returns:
            List[AutocompleteResult]: List of word completions.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata/stats.
        """
        pass

    def is_training_required(self) -> bool:
        """
        Whether this model requires training before use.

        Returns:
            bool: True for training-based models, False for pretrained/API-based models.
        """
        return True
