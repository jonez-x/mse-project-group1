import logging
import nltk
import pytest
import time
import pickle
from nltk.corpus import reuters
from typing import List

try:
    from src.retrieval_pipeline.dense_retriever import DenseRetriever
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from retrieval_pipeline.dense_retriever import DenseRetriever


def _load_reuters():
    """Ensure the Reuters corpus is available."""
    try:
        nltk.data.find("corpora/reuters")
    except LookupError:
        nltk.download("reuters")


LOGGER = logging.getLogger(name=__name__)


@pytest.fixture(scope="session")
def reuters_docs() -> List[str]:
    """Load a subset of Reuters corpus documents for testing."""
    _load_reuters()
    files = reuters.fileids()[:10000]
    return [reuters.raw(fileids=fid) for fid in files]


def _log_results(query: str, hits: List[tuple], elapsed: float) -> None:
    """Log query results with timing information."""
    LOGGER.info("Query: %s | %.3f ms | top=%d", query, elapsed * 1000, len(hits))
    for rank, (idx, score, text) in enumerate(hits[:5], start=1):
        snippet = " ".join(text.split()[:25])
        LOGGER.info("%2d. doc[%d] (%.4f): %s…", rank, idx, score, snippet)


def test_dense_ranking(reuters_docs):
    """Test DenseRetriever ranking functionality with oil prices query."""
    retriever = DenseRetriever()
    retriever.fit(reuters_docs)

    start = time.perf_counter()
    results = retriever.retrieve("oil prices", top_k=20)
    duration = time.perf_counter() - start

    _log_results("oil prices", results, duration)

    assert any("oil" in text.lower() for _, _, text in results), "Expected 'oil' in top results"


def test_dense_pickle_roundtrip(reuters_docs, tmp_path):
    """Test serialization and deserialization of DenseRetriever."""
    retriever = DenseRetriever()
    retriever.fit(reuters_docs)

    pkl_path = tmp_path / "dense_reuters.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(retriever, f)

    with open(pkl_path, "rb") as f:
        loaded = pickle.load(f)

    for system, label in [(retriever, "orig"), (loaded, "loaded")]:
        start = time.perf_counter()
        results = system.retrieve("federal reserve", top_k=10)
        elapsed = time.perf_counter() - start
        _log_results(f"federal reserve ({label})", results, elapsed)

    assert retriever.retrieve("federal reserve", top_k=10) == loaded.retrieve("federal reserve", top_k=10)
