from typing import List, Dict, Any
import logging

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

    def suggest(
            self,
            query: str,
            n_suggestions: int = 3
    ) -> List[AutocompleteResult]:
        # For empty queries, return an empty list
        if not query:
            return []

        # Determine if we should complete a word or predict next word
        if query.endswith(' '):
            # Predict next word
            words = query.lower().strip().split()
            if words:
                return self._predict_next_word(words[-1], n_suggestions)
        else:
            # Complete current word
            words = query.lower().strip().split()
            if words:
                current_word = words[-1]
                if len(current_word) >= 2:
                    return self._complete_word(current_word, n_suggestions)

        return []

    def _predict_next_word(
            self,
            last_word: str,
            max_suggestions: int = 3,
    ) -> List[AutocompleteResult]:
        """
        Predict the next word that commonly follow the given last word.

        Args:
            last_word (str): The last word in the query.
            max_suggestions (int): Maximum number of suggestions to return (default: 3).

        Returns:
            List[AutocompleteResult]: List of suggested next words with their scores.
        """
        try:
            response = requests.get(
                url=f"{self.BASE_URL}/words",
                params={
                    "rek_trg": last_word,
                    "max": max_suggestions,
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
                    ) for result in results
                ]

                # If no results with rel_trg, try rel_bga (bigram analysis)
                if not suggestions:
                    logger.info(f"No next word suggestions found for '{last_word}', trying bigram analysis.")

                    response = requests.get(
                        url=f"{self.BASE_URL}/words",
                        params={
                            "rel_bga": last_word,
                            "max": max_suggestions,
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
                            ) for result in results
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
            max_suggestions: int = 3,
    ) -> List[AutocompleteResult]:
        """
        Get word completions for a partial word.

        Args:
            partial_word (str): The partial word to complete.
            max_suggestions (int): Maximum number of suggestions to return (default: 3).

        Returns:
            List[AutocompleteResult]: List of suggested completions with their scores.
        """
        try:
            response = requests.get(
                url=f"{self.BASE_URL}/sug",
                params={
                    "s": partial_word,
                    "max": max_suggestions,
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
                    ) for result in results
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
    print(f"Predicting completions for 'hel':")
    suggestions_hel = model.suggest("hel")
    for suggestion in suggestions_hel:
        print(suggestion)

    print(f"\nPredicting next words for 'hello ':")
    suggestions_hello = model.suggest("hello ")
    for suggestion in suggestions_hello:
        print(suggestion)

    print(f"\nPredicting next words for 'hello world ':")
    suggestions_hello_world = model.suggest("hello world ")
    for suggestion in suggestions_hello_world:
        print(suggestion)
