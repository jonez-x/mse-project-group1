from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent
AUTOCOMPLETE_SYSTEM_DIR = PROJECT_ROOT / "autocomplete_system"

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = ["http://localhost:5173"]

# Database paths
DUCKDB_V1_PATH = PROJECT_ROOT / "crawler" / "tuebingen_crawl.duckdb"
DUCKDB_V2_PATH = PROJECT_ROOT / "crawler" / "crawler_2" / "data.db"

# Autocomplete Configuration
DEFAULT_AUTOCOMPLETE_MODEL = "datamuse"  # "ngram" or "datamuse"
MAX_AUTOCOMPLETE_SUGGESTIONS = 10
AUTOCOMPLETE_MIN_FREQUENCY = 2
AUTOCOMPLETE_NGRAM_ORDER = 2

# Model paths
TRAINED_MODELS_DIR = AUTOCOMPLETE_SYSTEM_DIR / "models" / "trained_models"
NGRAM_MODEL_PATH = TRAINED_MODELS_DIR / "ngram.pkl"

# Search Engine Configuration
USE_PRF = True
USE_RERANK = True

# Logging Configuration
LOG_LEVEL = "INFO"