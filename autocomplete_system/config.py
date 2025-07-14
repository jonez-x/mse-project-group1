from pathlib import Path

BASEDIR = Path(__file__).parent

# DUCKDB_PATH = BASEDIR / "data" / "autocomplete.duckdb"
DUCKDB_PATH = BASEDIR / "data" / "training_data.db"
SERIALIZED_DIR = BASEDIR / "data" / "serialized"
DEFAULT_MODEL = "ngram"  # "ngram", "trie", "hybrid"
MAX_SUGGESTIONS = 10
MIN_FREQUENCY = 2
NGRAM_ORDER = 3
