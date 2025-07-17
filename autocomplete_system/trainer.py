import pickle
from typing import Optional, Dict, Any, List

from models import NgramModel, AutocompleteModel
from config import *
from autocomplete_system.data import DataLoader

MODEL_CLASSES = {
    "ngram": lambda: NgramModel(NGRAM_ORDER),
    # "trie": TrieModel,
    # "hybrid": lambda : HybridModel(NGRAM_ORDER),
}


def train_and_serialize_model(
        model_name=DEFAULT_MODEL,
        texts: Optional[List[str]] = None,
        verbose: bool = True,
):
    texts = texts or DataLoader.load_duckdb_data(duckdb_path=(str(DUCKDB_PATH)))

    m: AutocompleteModel = MODEL_CLASSES[model_name]()

    m.train(
        texts=texts,
        min_freq=MIN_FREQUENCY,
    )

    path = SERIALIZED_DIR / f"{model_name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(m, f)

    if verbose:
        print(f"Model {model_name} trained and serialized to {path}")

    return m


def load_model(
        model_name: str = DEFAULT_MODEL,
        verbose: bool = True,
) -> AutocompleteModel:
    path = SERIALIZED_DIR / f"{model_name}.pkl"

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(path, "rb") as f:
        model: AutocompleteModel = pickle.load(f)

    if verbose:
        print(f"Model {model_name} loaded from {path}")

    return model

