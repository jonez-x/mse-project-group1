from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


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
            **kwargs: Dict[str, Any],
    ) -> None:
        """
        Build internal data structures.

        For API-based models, this method may be a no-op.
        For training-based models, this should process the texts.
        """
        pass

    @abstractmethod
    def suggest(
            self,
            query: str,
            n_suggestions: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Get autocomplete suggestions for the given query.

        Args:
            query (str): Input text to get suggestions for.
            n_suggestions (int): Maximum number of suggestions to return (default: 3).

        Returns:
            List[Tuple[str, float]]: List of tuples containing suggested words and their scores.
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
