import logging
import nltk
import pytest
from nltk.corpus import reuters
from typing import List, Union, Tuple
from retrieval_engine.docs.document_store import Document


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
def reuters_docs() -> List[Document]:
    """
    Load a subset of Reuters corpus documents for testing.

    Returns:
        List[Document]: A list of Document objects from the Reuters corpus.
    """
    load_reuters()
    files = reuters.fileids()[:10000]
    documents = []
    for i, fid in enumerate(files):
        raw_text = reuters.raw(fileids=fid)
        # Create Document objects with the raw text as both title and excerpt
        # This ensures to_text() returns the full content
        doc = Document(
            url=f"reuters://{fid}",
            title=f"Reuters Document {i}",
            excerpt=raw_text.strip(),
            main_image=None,
            favicon=None,
        )
        documents.append(doc)
    return documents


def log_query_results(
        query: str,
        results: Union[
            List[Document],  # BM25 format (Document objects)
            Tuple[List[int], List[float]],  # Old BM25 format
            List[Tuple[str, float, Document]],  # Dense format
            List[Tuple[str, float]]  # RetrievalEngine format
        ],
        elapsed: float,
        logger: logging.Logger,
        docs: List[Document] = None
):
    """
    Log query results with timing information for both BM25 and dense retrievers.

    Args:
        query (str): The search query string.
        results: The search results.
        elapsed (float): Time taken to execute the query in seconds.
        logger (logging.Logger): Logger instance to log the results.
        docs (List[Document], optional): Document objects for doc lookup (required for str ID format).
    """
    logger.info("Query: %s | %.3f ms | top=%d", query, elapsed * 1000, len(results))

    # BM25 format: List[Document]
    if isinstance(results, list) and results and isinstance(results[0], Document):
        for rank, doc in enumerate(results[:5], start=1):
            snippet = " ".join(doc.to_text().split()[:25])
            logger.info("%2d. %s: %s…", rank, doc.url, snippet)

    # Old BM25 format: (indices, scores)
    if isinstance(results, tuple) and len(results) == 2:
        indices, scores = results
        for rank, idx in enumerate(indices[:5], start=1):
            if docs:
                snippet = " ".join(docs[idx].to_text().split()[:25])
                score = scores[idx] if idx < len(scores) else 0.0
                logger.info("%2d. doc[%d] (%.4f): %s…", rank, idx, score, snippet)
            else:
                logger.info("%2d. doc[%d]", rank, idx)

    # Dense format: (idx, score, Document)
    elif isinstance(results, list) and results and isinstance(results[0], tuple):
        first = results[0]
        if len(first) == 3 and isinstance(first[2], Document):
            for rank, (doc_id, score, doc) in enumerate(results[:5], start=1):
                snippet = " ".join(doc.to_text().split()[:25])
                logger.info("%2d. doc[%s] (%.4f): %s…", rank, doc_id, score, snippet)
        # Legacy dense format: (idx, score, text)
        elif len(first) == 3 and isinstance(first[2], str):
            for rank, (idx, score, text) in enumerate(results[:5], start=1):
                snippet = " ".join(text.split()[:25])
                logger.info("%2d. doc[%d] (%.4f): %s…", rank, idx, score, snippet)

        # RetrievalEngine format: (str_doc_id, score)
        elif len(first) == 2 and isinstance(first[0], str):
            for rank, (doc_id, score) in enumerate(results[:5], start=1):
                try:
                    idx = int(doc_id)
                    snippet = " ".join(docs[idx].to_text().split()[:25]) if docs else ""
                    logger.info("%2d. doc[%s] (%.4f): %s…", rank, doc_id, score, snippet)
                except (ValueError, IndexError):
                    logger.info("%2d. doc[%s] (%.4f)", rank, doc_id, score)
