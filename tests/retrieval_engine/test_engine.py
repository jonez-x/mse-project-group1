import logging
import time
import pytest
from retrieval_engine.core.engine import RetrievalEngine
from test_utils import reuters_docs, log_query_results
import importlib.util

skip_if_no_torch = importlib.util.find_spec("torch") is None

def test_retrieval_engine_basic(reuters_docs):
    """End‑to‑end test: BM25 + Dense + RRF (ohne PRF / ReRank)."""
    engine = RetrievalEngine(
        use_prf=False,
        use_rerank=False,
    )
    engine.fit(reuters_docs)

    start = time.perf_counter()
    hits = engine.search("oil prices", bm25_top_k=300, dense_top_k=300, final_top_k=20)
    elapsed = time.perf_counter() - start

    LOGGER = logging.getLogger(__name__)
    log_query_results("oil prices", hits, elapsed, LOGGER, docs=reuters_docs)


    # Assert we got 20 hits and they are sorted by score descending
    assert len(hits) == 20
    assert all(hits[i][1] >= hits[i + 1][1] for i in range(len(hits) - 1))

    # Check that at least one of the top docs mentions the query term "oil"
    assert any("oil" in reuters_docs[int(idx)].lower() for idx, _ in hits), "Expected 'oil' in top results"


def test_prf_refine(reuters_docs):
    engine = RetrievalEngine(use_prf=True, use_rerank=False)
    engine.fit(reuters_docs[:2000])
    hits = engine.search("oil prices", final_top_k=5)
    assert any("oil" in reuters_docs[int(idx)].lower() for idx, _ in hits), "Expected 'oil' in top results"

@pytest.mark.skipif(skip_if_no_torch, reason="torch not installed")
def test_retrieval_engine_full(reuters_docs):
    """Full pipeline with PRF + Cross‑Encoder (slow, heavy)."""
    engine = RetrievalEngine(
        use_prf=True,
        use_rerank=True,
    )
    engine.fit(reuters_docs[:5000])  # speed!

    start = time.perf_counter()
    hits = engine.search("european central bank", final_top_k=100)
    elapsed = time.perf_counter() - start

    LOGGER = logging.getLogger(__name__)
    log_query_results("european central bank", hits, elapsed, LOGGER, docs=reuters_docs)

    assert any("bank" in reuters_docs[int(idx)].lower() for idx, _ in hits), "Expected 'bank' in top results"
