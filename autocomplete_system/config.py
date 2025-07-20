from pathlib import Path

BASEDIR = Path(__file__).parent
PROJECT_ROOT = BASEDIR.parent

# DUCKDB_PATH = BASEDIR / "data" / "autocomplete.duckdb"
DUCKDB_PATH = PROJECT_ROOT / "crawler" / "tuebingen_crawl.duckdb"
SERIALIZED_DIR = BASEDIR / "models" / "trained_models"
DEFAULT_MODEL = "ngram"  # "ngram", "trie", "hybrid"
MAX_SUGGESTIONS = 10
MIN_FREQUENCY = 2
NGRAM_ORDER = 2
