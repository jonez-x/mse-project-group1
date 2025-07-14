import logging
import nltk
import pytest
import time
from nltk.corpus import reuters
from typing import List, Tuple
from retrieval_engine.bm25_retriever import BM25Retriever

def _load_reuters():
    """
    Helper function to ensure the Reuters corpus is available.
    Loads the corpus if it is not already downloaded.
    """
    try:
        nltk.data.find("corpora/reuters")
    except LookupError:
        nltk.download(corpus="reuters")


# Initialize the logger
LOGGER = logging.getLogger(name=__name__)


@pytest.fixture(scope="session")
def reuters_docs() -> List[str]:
    """Load a subset of Reuters corpus documents for testing."""
    files = reuters.fileids()[:10000]
    return [reuters.raw(fileids=fid) for fid in files]


def _log_results(query: str, hits: Tuple[list[int], list[float]], docs: List[str], elapsed: float) -> None:
    """Log query results with timing information."""
    LOGGER.info("Query: %s | %.3f ms | top=%d", query, elapsed * 1000, len(hits[0]))
    for rank, idx in enumerate(iterable=hits[0][:5], start=1):
        snippet = " ".join(docs[idx].split()[:25])
        LOGGER.info("%2d. doc[%d]: %s…", rank, idx, snippet)


def test_bm25_ranking(reuters_docs):
    """Test BM25 indexer ranking functionality with oil prices query."""
    # Initialize and fit the BM25 indexer to the Reuters corpus
    bm25 = BM25Retriever()
    bm25 = bm25.fit(docs=reuters_docs)

    # Query for oil prices and measure performance
    start = time.perf_counter()
    hits = bm25.query(text="oil prices", top_k=20)
    duration = time.perf_counter() - start

    _log_results(query="oil prices", hits=hits, docs=reuters_docs, elapsed=duration)

    # Verify that results contain relevant documents
    assert any("oil" in reuters_docs[idx].lower() for idx in hits[0]), "Expected 'oil' in top results"


def test_pickle_roundtrip(reuters_docs, tmp_path):
    """Test serialization and deserialization of BM25 indexer."""
    # Train indexer and save to disk
    bm25 = BM25Retriever()
    bm25 = bm25.fit(docs=reuters_docs)

    pkl_path = tmp_path / "bm25_reuters.pkl"
    bm25.save(path=pkl_path)
    loaded = BM25Retriever.load(path=pkl_path)

    # Test both original and loaded indexers
    for retriever, label in [(bm25, "orig"), (loaded, "loaded")]:
        start = time.perf_counter()
        hits = retriever.query(text="federal reserve", top_k=10)
        elapsed = time.perf_counter() - start
        _log_results(query=f"federal reserve ({label})", hits=hits, docs=reuters_docs, elapsed=elapsed)

    # Verify identical results from both indexers
    assert bm25.query(text="federal reserve", top_k=10) == loaded.query(text="federal reserve", top_k=10)
