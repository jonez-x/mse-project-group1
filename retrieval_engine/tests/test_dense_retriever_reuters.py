import logging
import nltk
import pytest
import time
import pickle
from nltk.corpus import reuters
from typing import List, Tuple
from retrieval_engine.dense_retriever import DenseRetriever

LOGGER = logging.getLogger(__name__)

def _load_reuters() -> None:
    """Ensure the Reuters corpus is available locally."""
    try:
        nltk.data.find("corpora/reuters")
    except LookupError:
        nltk.download("reuters")


@pytest.fixture(scope="session")
def reuters_docs() -> List[str]:
    """Return a list of raw Reuters documents (subset) for testing."""
    _load_reuters()
    files = reuters.fileids()[:10000]  # keep runtime & memory reasonable
    return [reuters.raw(fid) for fid in files]


def _log_results(
    query: str,
    hits: List[Tuple[int, float]],
    elapsed: float,
    docs: List[str],
) -> None:
    """Log the first few hits for manual inspection during test runs."""
    LOGGER.info("Query: %s | %.3f ms | top=%d", query, elapsed * 1000, len(hits))
    for rank, (idx, score) in enumerate(hits[:5], start=1):
        snippet = " ".join(docs[idx].split()[:25])
        LOGGER.info("%2d. doc[%d] (%.4f): %s…", rank, idx, score, snippet)


def test_dense_ranking(reuters_docs):
    """Validate that semantically relevant documents are retrieved."""
    retriever = DenseRetriever()
    retriever.fit(reuters_docs)

    start = time.perf_counter()
    results = retriever.query("oil prices", top_k=20)
    duration = time.perf_counter() - start

    _log_results("oil prices", results, duration, reuters_docs)

    # Ensure at least one of the returned docs actually contains the keyword.
    assert any("oil" in reuters_docs[idx].lower() for idx, _ in results), (
        "Expected at least one document mentioning 'oil' in top results",
    )


def test_dense_pickle_roundtrip(reuters_docs, tmp_path):
    """Check that a fitted DenseRetriever can be pickled/unpickled loss‑lessly."""
    retriever = DenseRetriever()
    retriever.fit(reuters_docs)

    pkl_path = tmp_path / "dense_reuters.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(retriever, f)

    with open(pkl_path, "rb") as f:
        loaded = pickle.load(f)

    for system, label in [(retriever, "orig"), (loaded, "loaded")]:
        start = time.perf_counter()
        results = system.query("federal reserve", top_k=10)
        elapsed = time.perf_counter() - start
        _log_results(f"federal reserve ({label})", results, elapsed, reuters_docs)

    # The rankings (indices + scores) should be identical after the round‑trip.
    assert retriever.query("federal reserve", top_k=10) == loaded.query(
        "federal reserve", top_k=10
    )
