"""
Tests for DenseRetriever using the Reuters corpus dataset.
"""
import logging
import pickle
import pytest
import time
from pathlib import Path
from typing import List, Sequence

from test_utils import reuters_docs, log_query_results
from retrieval_engine.dense_retriever import DenseRetriever

# Initialize the logger
LOGGER = logging.getLogger(name=__name__)


def test_dense_ranking(
        reuters_docs: List[str],
        search_term: str = "oil"
) -> None:
    """Test DenseRetriever ranking functionality with oil prices query."""
    # Initialize and fit the DenseRetriever to the Reuters corpus
    retriever = DenseRetriever()
    retriever.fit(corpus=reuters_docs)

    query = f"{search_term} prices"

    # Query for oil prices and measure performance  
    start = time.perf_counter()
    results = retriever.query(
        query=query,
        top_k=20
    )
    duration = time.perf_counter() - start

    # Log the query results
    log_query_results(
        query=query,
        results=results,
        elapsed=duration,
        logger=LOGGER
    )

    # Verify that results contain relevant documents
    assert any(search_term in text.lower() for _, _, text in results), f"Expected '{search_term}' in top results"


@pytest.mark.parametrize("search_term", [
    # "oil",
    "gold",
    "stock",
    "bank",
    "trade"
])
def test_dense_ranking_multiple_terms(
        reuters_docs: List[str],
        search_term: str,
) -> None:
    """Test DenseRetriever ranking functionality with multiple search terms."""
    test_dense_ranking(
        reuters_docs=reuters_docs,
        search_term=search_term
    )


def test_dense_pickle_roundtrip(
        reuters_docs: List[str],
        tmp_path: Path
) -> None:
    """Test serialization and deserialization of DenseRetriever."""
    # Train retriever and save to disk
    retriever = DenseRetriever()
    retriever.fit(corpus=reuters_docs)

    pkl_path = tmp_path / "dense_reuters.pkl"

    # Save and load the retriever using pickle
    with open(pkl_path, "wb") as f:
        pickle.dump(retriever, f)

    with open(pkl_path, "rb") as f:
        loaded = pickle.load(f)

    # Test both original and loaded retrievers
    for system, label in [(retriever, "orig"), (loaded, "loaded")]:
        start = time.perf_counter()
        results = system.retrieve(
            query="federal reserve",
            top_k=10
        )
        elapsed = time.perf_counter() - start

        log_query_results(
            query=f"federal reserve ({label})",
            results=results,
            elapsed=elapsed,
            logger=LOGGER
        )

    # Verify identical results from both retrievers
    orig_results = retriever.query(query="federal reserve", top_k=10)
    loaded_results = loaded.query(query="federal reserve", top_k=10)

    # Compare results with tolerance for floating point precision
    assert len(orig_results) == len(loaded_results), "Result counts should match"
    for (orig_idx, orig_score, orig_text), (loaded_idx, loaded_score, loaded_text) in zip(orig_results, loaded_results):
        assert orig_idx == loaded_idx, f"Document indices should match: {orig_idx} vs {loaded_idx}"
        assert abs(orig_score - loaded_score) < 1e-6, f"Scores should be very close: {orig_score} vs {loaded_score}"
        assert orig_text == loaded_text, f"Document texts should match"
