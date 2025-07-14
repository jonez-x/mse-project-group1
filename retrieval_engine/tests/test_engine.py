import logging
import time
from typing import List, Tuple

import nltk
import pytest
from nltk.corpus import reuters
from retrieval_engine.retrieval_engine import RetrievalEngine

LOGGER = logging.getLogger(__name__)


def _load_reuters():
    """Ensure Reuters corpus is present (download once)."""
    try:
        nltk.data.find("corpora/reuters")
    except LookupError:
        nltk.download("reuters")


@pytest.fixture(scope="session")
def reuters_docs() -> List[str]:
    """Return a manageable subset (10 k docs) of Reuters raw texts."""
    _load_reuters()
    files = reuters.fileids()[:10000]
    return [reuters.raw(fid) for fid in files]


def _log_results(query: str, hits: List[Tuple[int, float]], docs: List[str], elapsed: float) -> None:
    LOGGER.info("Query: %s | %.1f ms | top=%d", query, elapsed * 1e3, len(hits))
    preview = hits[:5]
    for rank, (idx, score) in enumerate(preview, 1):
        snippet = " ".join(docs[idx].split()[:20])
        LOGGER.info("%2d. doc[%d] %.3f → %s…", rank, idx, score, snippet)



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

    _log_results("oil prices", hits, reuters_docs, elapsed)

    # Assert we got 20 hits and they are sorted by score descending
    assert len(hits) == 20
    assert all(hits[i][1] >= hits[i + 1][1] for i in range(len(hits) - 1))

    # Check that at least one of the top docs mentions the query term "oil"
    assert any("oil" in reuters_docs[idx].lower() for idx, _ in hits), "Expected 'oil' in top results"


@pytest.mark.skipif("torch" not in globals(), reason="Slow – requires GPU/CPU transformers model")
def test_retrieval_engine_full(reuters_docs):
    """Full pipeline with PRF + Cross‑Encoder (slow, heavy)."""
    engine = RetrievalEngine(
        use_prf=True,
        use_rerank=True,
    )
    engine.fit(reuters_docs[:5000])  # speed!

    start = time.perf_counter()
    hits = engine.search("european central bank", final_top_k=10)
    elapsed = time.perf_counter() - start

    _log_results("european central bank", hits, reuters_docs, elapsed)

    assert hits, "Engine returned no results"
