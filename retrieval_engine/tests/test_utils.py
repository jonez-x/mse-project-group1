import logging
import nltk
import pytest
from nltk.corpus import reuters
from typing import List, Union, Tuple


def load_reuters() -> None:
    """
    Ensure the Reuters corpus is available for testing.
    Downloads the corpus if it is not already present.
    """
    try:
        nltk.data.find("corpora/reuters")
    except LookupError:
        nltk.download("reuters")


@pytest.fixture(scope="session")
def reuters_docs() -> List[str]:
    """
    Load a subset of Reuters corpus documents for testing.

    Returns:
        List[str]: A list of raw text documents from the Reuters corpus.
    """
    load_reuters()
    files = reuters.fileids()[:10000]
    return [reuters.raw(fileids=fid) for fid in files]


def log_query_results(
        query: str,
        results: Union[
            Tuple[List[int], List[float]],  # BM25 format: (indices, scores)
            List[Tuple[int, float, str]]  # Dense format: [(idx, score, text), ...]
        ],
        elapsed: float,
        logger: logging.Logger,
        docs: List[str] = None
):
    """
    Log query results with timing information for both BM25 and dense retrievers.

    Args:
        query (str): The search query string.
        results (Union[Tuple[List[int], List[float]], List[Tuple[int, float, str]]]):
            The search results in either BM25 format ((indices, scores)) or dense format ([(idx, score, text), ...]).
        elapsed (float): Time taken to execute the query in seconds.
        logger (logging.Logger): Logger instance to log the results.
        docs (List[str], optional): List of documents for BM25 format results. Defaults to None.
    """

    # Handle BM25 format: (indices, scores) tuple
    if isinstance(results, tuple) and len(results) == 2:
        indices, scores = results
        result_count = len(indices)
        logger.info("Query: %s | %.3f ms | top=%d", query, elapsed * 1000, result_count)

        # Show top 5 results
        for rank, idx in enumerate(indices[:5], start=1):
            if docs:
                snippet = " ".join(docs[idx].split()[:25])
                if scores and len(scores) > 0:
                    logger.info("%2d. doc[%d] (%.4f): %s…",
                                rank, idx, scores[idx] if idx < len(scores) else 0.0, snippet)
                else:
                    logger.info("%2d. doc[%d]: %s…", rank, idx, snippet)

    # Handle dense format: [(idx, score, text), ...] list
    elif isinstance(results, list) and results and isinstance(results[0], tuple):
        result_count = len(results)
        logger.info("Query: %s | %.3f ms | top=%d", query, elapsed * 1000, result_count)

        # Show top 5 results
        for rank, (idx, score, text) in enumerate(results[:5], start=1):
            snippet = " ".join(text.split()[:25])
            logger.info("%2d. doc[%d] (%.4f): %s…",
                        rank, idx, score, snippet)
