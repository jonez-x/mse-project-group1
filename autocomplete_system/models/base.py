from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class AutocompleteModel(ABC):
    """
    Abstract base class for autocompletion models.
    """
    name: str

    @abstractmethod
    def train(
            self,
            texts: List[str],
            **kwargs: Dict[str, Any],
    ) -> None:
        """
        Build internal data structures.
        """
        pass

    @abstractmethod
    def suggest(
            self,
            query: str,
            n_suggestions: int = 3,
    ) -> List[Tuple[str, float]]:
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata/stats.
        """
        pass
