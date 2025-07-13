"""
Tests for BM25Indexer using the Reuters corpus dataset.
"""
import logging
import time
from pathlib import Path
from typing import List

from test_utils import reuters_docs, log_query_results

# TODO: find a workaround for the relative import issue
try:
    from src.retrieval_pipeline.bm25_indexer import BM25Indexer
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from retrieval_pipeline.bm25_indexer import BM25Indexer

# Initialize the logger
LOGGER = logging.getLogger(name=__name__)


def test_bm25_ranking(reuters_docs: List[str]):
    """Test BM25 indexer ranking functionality with oil prices query."""
    # Initialize and fit the BM25 indexer to the Reuters corpus
    bm25 = BM25Indexer()
    bm25 = bm25.fit(docs=reuters_docs)

    # Query for oil prices and measure performance
    start = time.perf_counter()
    hits = bm25.query(
        text="oil prices",
        top_n=20
    )
    duration = time.perf_counter() - start

    # Log the query results
    log_query_results(
        query="oil prices",
        results=hits,
        elapsed=duration,
        logger=LOGGER,
        docs=reuters_docs
    )

    # Verify that results contain relevant documents
    assert any("oil" in reuters_docs[idx].lower() for idx in hits[0]), "Expected 'oil' in top results"


def test_pickle_roundtrip(
        reuters_docs: List[str],
        tmp_path: Path,
):
    """Test serialization and deserialization of BM25 indexer."""
    # Train indexer and save to disk
    bm25 = BM25Indexer()
    bm25 = bm25.fit(docs=reuters_docs)

    pkl_path = tmp_path / "bm25_reuters.pkl"
    bm25.save(path=pkl_path)
    loaded = BM25Indexer.load(path=pkl_path)

    # Test both original and loaded indexers
    for retriever, label in [(bm25, "orig"), (loaded, "loaded")]:
        start = time.perf_counter()
        hits = retriever.query(text="federal reserve", top_n=10)
        elapsed = time.perf_counter() - start

        log_query_results(
            query=f"federal reserve ({label})",
            results=hits,
            elapsed=elapsed,
            logger=LOGGER,
            docs=reuters_docs
        )

    # Verify identical results from both indexers
    assert bm25.query(text="federal reserve", top_n=10) == loaded.query(text="federal reserve", top_n=10)
