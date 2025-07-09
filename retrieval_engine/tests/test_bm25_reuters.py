import logging
import time
from typing import List

import pytest
import nltk
from nltk.corpus import reuters

from retrieval_pipeline.bm25_indexer import BM25Indexer

try:
    nltk.data.find("corpora/reuters.zip")
except LookupError:
    nltk.download("reuters")


LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def reuters_docs() -> List[str]:
    files = reuters.fileids()[:10000]
    return [reuters.raw(fid) for fid in files]


def _log_results(query: str, hits: List[int], docs: List[str], elapsed: float) -> None:
    LOGGER.info("Query: %s | %.3f ms | top=%d", query, elapsed * 1000, len(hits))
    for rank, idx in enumerate(hits[0][:5], 1):
        snippet = " ".join(docs[idx].split()[:25])
        LOGGER.info("%2d. doc[%d]: %s…", rank, idx, snippet)


def test_bm25_ranking(reuters_docs):
    bm25 = BM25Indexer().fit(reuters_docs)

    start = time.perf_counter()
    hits = bm25.query("oil prices", top_n=20)
    duration = time.perf_counter() - start

    _log_results("oil prices", hits, reuters_docs, duration)

    assert any("oil" in reuters_docs[idx].lower() for idx in hits[0]), "Expected 'oil' in top results"


def test_pickle_roundtrip(reuters_docs, tmp_path):
    bm25 = BM25Indexer().fit(reuters_docs)

    pkl_path = tmp_path / "bm25_reuters.pkl"
    bm25.save(pkl_path)
    loaded = BM25Indexer.load(pkl_path)

    for retriever, label in [(bm25, "orig"), (loaded, "loaded")]:
        start = time.perf_counter()
        hits = retriever.query("federal reserve", 10)
        elapsed = time.perf_counter() - start
        _log_results(f"federal reserve ({label})", hits, reuters_docs, elapsed)

    assert bm25.query("federal reserve", 10) == loaded.query("federal reserve", 10)
