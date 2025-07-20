from typing import List, Dict, Any
import logging
import re

import requests

from autocomplete_system.models.base import AutocompleteModel, AutocompleteResult

logger = logging.getLogger(__name__)


class DataMuseModel(AutocompleteModel):
    """
    DataMuse API-based autocompletion model.

    Implements the AutocompleteModel interface to provide word completion and
    next-word suggestions using the DataMuse API.

    This model does not require training and uses the DataMuse API for suggestions.

    Attributes:
        api_url (str): The base URL for the DataMuse API (set to "https://api.datamuse.com/sug" by default).
        REQUEST_TIMEOUT (float): Timeout for API requests in seconds (default: 1.0).
    """

    BASE_URL = "https://api.datamuse.com"
    REQUEST_TIMEOUT = 1.0  # seconds

    def __init__(self):
        super().__init__(name="datamuse")

    def is_training_required(self) -> bool:
        """DataMuse model does not require training."""
        return False

    def train(
            self,
            texts: List[str],
            **kwargs: Dict[str, Any],
    ) -> None:
        """No-op for API-based models like DataMuse."""
        pass

    def _is_valid_word(self, word: str) -> bool:
        """
        Check if a word is valid (contains only letters, hyphens, and apostrophes).
        
        Args:
            word (str): The word to validate.
            
        Returns:
            bool: True if the word is valid, False otherwise.
        """
        # Only allow words with letters, hyphens, and apostrophes
        # This filters out punctuation like ".", ",", numbers, etc.
        return bool(re.match(r"^[a-zA-Z'-]+$", word)) and len(word) >= 2

    def _predict_next_word(
            self,
            words: List[str],
            n_suggestions: int = 3,
    ) -> List[AutocompleteResult]:
        """
        Predict the next word that commonly follow the given words.

        Args:
            words (List[str]): List of preceding words.
            n_suggestions (int): Maximum number of suggestions to return (default: 3).

        Returns:
            List[AutocompleteResult]: List of suggested next words with their scores.
        """
        last_word = words[-1]  # Use the last word for the API call
        try:
            response = requests.get(
                url=f"{self.BASE_URL}/words",
                params={
                    "rek_trg": last_word,
                    "max": n_suggestions,
                },
                timeout=self.REQUEST_TIMEOUT,
            )

            if response.status_code == 200:
                results: List[Dict[str, Any]] = response.json()

                suggestions: List[AutocompleteResult] = [
                    AutocompleteResult(
                        word=result["word"],
                        score=float(result.get("score", 1.0)),  # Default score if not provided
                        type="next_word"
                    ) for result in results if self._is_valid_word(result["word"])
                ]

                # If no results with rel_trg, try rel_bga (bigram analysis)
                if not suggestions:
                    logger.info(f"No next word suggestions found for '{last_word}', trying bigram analysis.")

                    response = requests.get(
                        url=f"{self.BASE_URL}/words",
                        params={
                            "rel_bga": last_word,
                            "max": n_suggestions,
                        },
                        timeout=self.REQUEST_TIMEOUT,
                    )
                    if response.status_code == 200:
                        results = response.json()
                        suggestions = [
                            AutocompleteResult(
                                word=result["word"],
                                score=float(result.get("score", 1.0)),
                                type="next_word"
                            ) for result in results if self._is_valid_word(result["word"])
                        ]

                return suggestions

            else:
                logger.warning(f"Error fetching next word suggestions: {response.status_code}")
                return []

        except requests.RequestException as e:
            logger.error(f"Error fetching next word suggestions: {e}")
            return []

    def _complete_word(
            self,
            partial_word: str,
            n_suggestions: int = 3,
    ) -> List[AutocompleteResult]:
        """
        Get word completions for a partial word.

        Args:
            partial_word (str): The partial word to complete.
            n_suggestions (int): Maximum number of suggestions to return (default: 3).

        Returns:
            List[AutocompleteResult]: List of suggested completions with their scores.
        """
        try:
            response = requests.get(
                url=f"{self.BASE_URL}/sug",
                params={
                    "s": partial_word,
                    "max": n_suggestions,
                },
                timeout=self.REQUEST_TIMEOUT,
            )

            if response.status_code == 200:
                results: List[Dict[str, Any]] = response.json()

                return [
                    AutocompleteResult(
                        word=result["word"],
                        score=float(result.get("score", 1.0)),  # Default score if not provided
                        type="completion"
                    ) for result in results if self._is_valid_word(result["word"])
                ]

            else:
                logger.warning(f"Error fetching word completions: {response.status_code}")
                return []

        except requests.RequestException as e:
            logger.error(f"Error fetching word completions: {e}")
            return []

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata/stats."""
        return {
            "name": self.name,
            "type": "api_based",
            "requires_training": False,
            "base_url": self.BASE_URL,
            "timeout": self.REQUEST_TIMEOUT
        }


# Example usage:
if __name__ == "__main__":
    model = DataMuseModel()

    # Test the model with some example words
    example_words = ["food", "movie", "tuebingen", "car", "university"]
    for word in example_words:
        # Test word completion (partial prefix)
        suggestions = model.suggest(
            query=word[:3],
            n_suggestions=5,
        )
        print(f"\nSuggestions for '{word[:3]}': ")
        for suggestion in suggestions:
            print(f"  - {suggestion.word} (score: {suggestion.score})")

        # Test next word prediction (full word + space)
        next_word_suggestions = model.suggest(
            query=word + " ",
            n_suggestions=5,
        )
        print(f"\nNext word suggestions after '{word}': ")
        for suggestion in next_word_suggestions:
            print(f"  - {suggestion.word} (score: {suggestion.score}) [{suggestion.type}]")
