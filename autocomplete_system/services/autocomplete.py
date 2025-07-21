import logging
from typing import List, Optional, Dict, Any
from enum import Enum
import pickle
from pathlib import Path
import sys

from autocomplete_system.models import AutocompleteModel, DataMuseModel

logger = logging.getLogger(__name__)

NGRAM_PATH = Path(__file__).parent.parent / "models" / "trained_models" / "ngram.pkl"


class ModelType(Enum):
    """Available autocomplete models."""
    NGRAM = "ngram"
    DATAMUSE = "datamuse"


class AutocompleteService:
    """
    Autocomplete service that provides suggestions based on user input.

    Supports Ngram and DataMuse models for word completion and next-word suggestions.

    Attributes:
        models (Dict[ModelType, Optional[AutocompleteModel]]): Dictionary of available models
        default_model (ModelType): The default model to use for suggestions
    """

    def __init__(self, default_model: ModelType = ModelType.DATAMUSE) -> None:
        """
        Initialize the autocomplete service.

        Args:
            default_model (ModelType): The default model to use for suggestions (default: ModelType.DATAMUSE).
        """
        # Initialize models
        self.models: Dict[ModelType, Optional[AutocompleteModel]] = {
            ModelType.NGRAM: None,
            ModelType.DATAMUSE: None,
        }
        self.default_model = default_model
        self._initialize_models()

    def _initialize_models(self):
        """Initialize the available models."""
        try:
            # Initialize DataMuse model (always available)
            self.models[ModelType.DATAMUSE] = DataMuseModel()
            logger.info("DataMuse model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize DataMuse model: {e}")

        try:
            # Try to load trained ngram model
            if NGRAM_PATH.exists():
                with open(NGRAM_PATH, "rb") as f:
                    self.models[ModelType.NGRAM] = pickle.load(f)
                logger.info("Ngram model loaded successfully")
            else:
                logger.warning(f"Ngram model file not found at {NGRAM_PATH}")
                # If ngram is requested as default model, try to train it
                if self.default_model == ModelType.NGRAM:
                    logger.info("Attempting to train ngram model since it was requested...")
                    success = self._train_ngram_model()
                    if success:
                        logger.info("Successfully trained and loaded ngram model")
                    else:
                        logger.error("Failed to train ngram model, will fall back to DataMuse")
        except Exception as e:
            logger.warning(f"Failed to load ngram model: {e}")

    def get_suggestions(
            self,
            query: str,
            model_type: Optional[ModelType] = None,
            max_suggestions: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get autocomplete suggestions for the given query.

        Args:
            query (str): Input text to get suggestions for.
            model_type (Optional[ModelType]): Specific model to use for suggestions (default: None, uses default model).
            max_suggestions (int): Maximum number of suggestions to return (default: 5).

        Returns:
            List[Dict[str, Any]]: List of suggestions with word, score, type, model, and full_query.
        """
        if not query:
            return []

        # Use specified model or default
        model_to_use = model_type or self.default_model
        model = self.models.get(model_to_use)

        if not model:
            logger.error(f"Model {model_to_use} not available")
            return []

        try:
            # Get suggestions from model
            results = model.suggest(
                query=query,
                n_suggestions=max_suggestions,
            )

            # Convert to simple dictionary format
            suggestions = []
            for result in results:
                suggestion = {
                    "word": result.word,
                    "score": result.score,
                    "type": result.type,
                    "model": model_to_use.value
                }

                # Create full query for frontend
                if result.type == "completion":
                    # Replace the last word with the suggestion
                    words = query.split()
                    if words:
                        words[-1] = result.word
                        suggestion["full_query"] = " ".join(words)
                    else:
                        suggestion["full_query"] = result.word
                else:
                    # Next word - append to query
                    suggestion["full_query"] = query.rstrip() + " " + result.word

                suggestions.append(suggestion)

            return suggestions

        except Exception as e:
            logger.error(f"Error getting suggestions with {model_to_use}: {e}")
            return []

    def get_available_models(self) -> List[str]:
        """
        et list of available models.

        Returns:
            List[str]: List of model names that are currently available.
        """
        return [model_type.value for model_type, model in self.models.items() if model is not None]

    def get_model_info(self, model_type: ModelType) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.

        Args:
            model_type (ModelType): The type of model to get information for.

        Returns:
            Optional[Dict[str, Any]]: Dictionary with model information or None if model is not available.
        """
        model = self.models.get(model_type)
        if model:
            try:
                return model.get_model_info()
            except Exception as e:
                logger.error(f"Error getting model info for {model_type}: {e}")
        return None

    def _train_ngram_model(self) -> bool:
        """Train the ngram model if it doesn't exist.
        
        Returns:
            bool: True if training was successful, False otherwise.
        """
        try:
            # Import trainer function
            sys.path.append(str(Path(__file__).parent.parent.parent))
            from autocomplete_system.trainer import train_and_serialize_model
            
            # Create models directory if it doesn't exist
            NGRAM_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info("Training ngram model...")
            
            # Train the model
            train_and_serialize_model(
                model_name="ngram",
                texts=None,  # Will use default data loader
                verbose=True,
            )
            
            # Try to load the newly trained model
            if NGRAM_PATH.exists():
                with open(NGRAM_PATH, "rb") as f:
                    self.models[ModelType.NGRAM] = pickle.load(f)
                logger.info("Successfully trained and loaded ngram model")
                return True
            else:
                logger.error("Model training completed but file was not created")
                return False
                
        except Exception as e:
            logger.error(f"Failed to train ngram model: {e}")
            return False
