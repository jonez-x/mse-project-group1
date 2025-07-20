import pickle
from typing import Optional, Dict, Any, List

from models import NgramModel, AutocompleteModel
from config import *
from autocomplete_system.data import DataLoader

MODEL_CLASSES = {
    "ngram": lambda: NgramModel(n=NGRAM_ORDER),
    # "trie": TrieModel,
    # "hybrid": lambda : HybridModel(NGRAM_ORDER),
}


def train_and_serialize_model(
        model_name: str = DEFAULT_MODEL,
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


# Example usage:
if __name__ == "__main__":
    # Train and serialize the model
    trained_model = train_and_serialize_model(
        model_name="ngram",
        texts=None,  # Load texts from DuckDB if None
        verbose=True
    )

    # Load the model
    loaded_model = load_model(
        model_name="ngram",
        verbose=True
    )

    # Check if the loaded model is the same as the trained one
    assert isinstance(loaded_model, NgramModel)
    print("Model loaded successfully and is of type NgramModel.")

    # Debug: Check if n-grams were built
    print(f"N-gram counts: {[(k, len(loaded_model.ngrams[k])) for k in loaded_model.ngrams]}")

    # Test the model with some example words
    example_words = ["food", "movie", "tuebingen", "car", "university"]
    for word in example_words:
        # Test word completion (partial prefix)
        suggestions = loaded_model.suggest(
            query=word[:3],
            n_suggestions=5
        )
        print(f"\nSuggestions for '{word[:3]}': ")
        for suggestion in suggestions:
            print(f"  - {suggestion.word} (score: {suggestion.score})")

        # Test next word prediction (full word + space)
        next_word_suggestions = loaded_model.suggest(
            query=word + " ",
            n_suggestions=5
        )
        print(f"\nNext word suggestions after '{word}': ")
        for suggestion in next_word_suggestions:
            print(f"  - {suggestion.word} (score: {suggestion.score}) [{suggestion.type}]")
