import gzip
import os
from typing import List, Tuple, Union

import duckdb

from retrieval_engine.core.engine import RetrievalEngine
from retrieval_engine.docs.document_store import Document

INPUT_FILE = "queries/queries.txt"
OUTPUT_FILE = "results/results.txt"
DB_PATH = "../crawler/crawler_2/final/final.db"


def load_documents() -> List[Document]:
    """
    Load the documents from the DuckDB database.

    Returns:
        List[Document]: A list of Document objects containing the URL, title, excerpt, main image, and favicon.
    """
    con = duckdb.connect(DB_PATH)

    sql_query = """
                SELECT link, title, content, image_url
                FROM main.documents
                """
    rows = con.execute(sql_query).fetchall()

    documents: List[Document] = []
    for link, title, blob, image_url in rows:
        try:
            content = gzip.decompress(blob).decode("utf-8", errors="ignore")
        except Exception:
            content = ""

        excerpt = content[:300]
        doc = Document(
            url=link,
            title=title,
            excerpt=excerpt,
            main_image=image_url,
            favicon=None,
        )
        documents.append(doc)
    return documents


def run_batch_queries() -> None:
    """
    Run a batch of queries from a text file against the retrieval engine and save the results.

    Produces one ranked result per line as tab-separated entries:
    query_number \t rank_position \t document_url \t relevance_score
    """
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"File '{INPUT_FILE}' not found.")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    if not queries:
        raise ValueError("The input file contains no valid queries.")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Load and index documents
    docs = load_documents()
    engine = RetrievalEngine(use_prf=False, use_rerank=True)
    engine.fit(corpus=docs)
    print(f"Found {len(docs)} documents in the database.")

    # Open output file and write results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for q_idx, query in enumerate(queries, start=1):
            try:
                results: List[Union[Tuple[Document, float], Document]] = engine.search(query=query)
                for rank, entry in enumerate(results, start=1):
                    if isinstance(entry, tuple) and len(entry) == 2:
                        doc, score = entry
                    else:
                        doc = entry  # type: ignore
                        score = getattr(doc, 'score', None)
                    out.write(f"{q_idx}\t{rank}\t{doc.url}\t{score}\n")
            except Exception as e:
                print(f"Error processing query {q_idx}: {e}")

    print(f"Saved results to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    run_batch_queries()
