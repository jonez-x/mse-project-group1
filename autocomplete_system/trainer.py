import pickle
import sys
from pathlib import Path
from typing import List, Optional

from autocomplete_system.data import DataLoader
from autocomplete_system.models import AutocompleteModel, NgramModel

# Configuration
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DUCKDB_V1_PATH as DUCKDB_PATH,
    TRAINED_MODELS_DIR as SERIALIZED_DIR,
    AUTOCOMPLETE_MIN_FREQUENCY as MIN_FREQUENCY,
    AUTOCOMPLETE_NGRAM_ORDER as NGRAM_ORDER,
)

MODEL_CLASSES = {
    "ngram": lambda: NgramModel(n=NGRAM_ORDER),
    # Intentionally, one could have also implemented other models here (e.g., Trie or Hybrid approaches)
}


def train_and_serialize_model(
        model_name: str = "ngram",
        texts: Optional[List[str]] = None,
        verbose: bool = True,
):
    """
    Method to train a model (e.g., N-gram) and serialize it to disk.

    Calls the implemented train() method of the model class and saves the trained model to a file.

    Args:
        model_name (str): Name of the model to train and serialize (default: "ngram").
        texts (Optional[List[str]]): List of texts to train the model on. If None, loads from DuckDB.
        verbose (bool): Whether to print verbose output during training and serialization.
    """
    # If no texts are provided, load from DuckDB (default database)
    texts = texts or DataLoader.load_duckdb_data(duckdb_path=(str(DUCKDB_PATH)))

    # Instantiate the model class based on the model name
    m: AutocompleteModel = MODEL_CLASSES[model_name]()

    # Train the model with the provided texts
    m.train(
        texts=texts,
        min_freq=MIN_FREQUENCY,
    )

    # Make sure the folder exists
    path = SERIALIZED_DIR / f"{model_name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(m, f)

    if verbose:
        print(f"Model {model_name} trained and serialized to {path}")

    return m


def load_model(
        model_name: str = "ngram",
        verbose: bool = True,
) -> AutocompleteModel:
    """
    Method to load a serialized model from disk.

    Args:
        model_name (str): Name of the model to load (default: "ngram").
        verbose (bool): Whether to print verbose output during loading.

    Returns:
        AutocompleteModel: The loaded model instance.
    """
    path = SERIALIZED_DIR / f"{model_name}.pkl"

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(path, "rb") as f:
        model: AutocompleteModel = pickle.load(f)

    if verbose:
        print(f"Model {model_name} loaded from {path}")

    return model


if __name__ == "__main__":
    # Train and serialize the model
    train_and_serialize_model(
        model_name="ngram",
        texts=None,  # Load texts from DuckDB if None
        verbose=True
    )
