"""
Tests for BM25Indexer using the Reuters corpus dataset.
"""
import logging
import time
from pathlib import Path
from typing import List

from test_utils import reuters_docs, log_query_results
from retrieval_engine.retrievers.sparse import BM25Retriever

# Initialize the logger
LOGGER = logging.getLogger(name=__name__)


def test_bm25_ranking(reuters_docs: List):
    """Test BM25 indexer ranking functionality with oil prices query."""
    # Initialize and fit the BM25 indexer to the Reuters corpus
    bm25 = BM25Retriever()
    bm25 = bm25.fit(docs=reuters_docs)

    # Query for oil prices and measure performance
    start = time.perf_counter()
    hits = bm25.query(
        query="oil prices",
        top_k=20,
    )
    duration = time.perf_counter() - start

    # Log the query results
    log_query_results(
        query="oil prices",
        results=hits,
        elapsed=duration,
        logger=LOGGER,
        docs=reuters_docs,
    )

    # Verify that results contain relevant documents
    assert any("oil" in doc.to_text().lower() for doc in hits), "Expected 'oil' in top results"


def test_pickle_roundtrip(
        reuters_docs: List,
        tmp_path: Path,
):
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
        hits = retriever.query(
            query="federal reserve",
            top_k=10,
        )
        elapsed = time.perf_counter() - start

        log_query_results(
            query=f"federal reserve ({label})",
            results=hits,
            elapsed=elapsed,
            logger=LOGGER,
            docs=reuters_docs,
        )

    # Verify identical results from both indexers (compare document URLs)
    orig_results = bm25.query(
        query="federal reserve",
        top_k=10,
    )
    loaded_results = loaded.query(
        query="federal reserve",
        top_k=10,
    )
    orig_urls = [doc.url for doc in orig_results]
    loaded_urls = [doc.url for doc in loaded_results]
    assert orig_urls == loaded_urls
