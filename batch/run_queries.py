import gzip
import os
from typing import List

import duckdb

from retrieval_engine.core.engine import RetrievalEngine
from retrieval_engine.docs.document_store import Document

INPUT_FILE = "queries/queries.txt"
OUTPUT_FILE = "results/results.txt"
DB_PATH = "../crawler/crawler_2/final/data.db"


def load_documents() -> List[Document]:
    """
    Load the documents from the DuckDB database.

    Returns:
        List[Document]: A list of Document objects containing the URL, title, excerpt, main image, and favicon.
    """
    con = duckdb.connect(DB_PATH)

    sql_query = """
                SELECT link, title, content, image_url
                FROM main.documents_filtered3 \
                """
    rows = con.execute(sql_query).fetchall()

    documents = []
    for link, title, blob, image_url in rows:
        try:
            content = gzip.decompress(blob).decode("utf-8", errors="ignore")
        except Exception:
            content = ""

        # Only look at the first 300 characters for the excerpt
        excerpt = content[:300]
        doc = Document(
            url=link,
            title=title,
            excerpt=excerpt,
            main_image=image_url,
            favicon=None,  # Favicon is not available here
        )
        documents.append(doc)
    return documents


def run_batch_queries() -> None:
    """
    Run a batch of queries from a text file against the retrieval engine and save the results.

    Uses the queries from 'queries/queries.txt' and saves the results to 'results/results.txt'.

    """
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"File '{INPUT_FILE}' not found.")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    # Filter out empty queries
    if not queries:
        raise ValueError("The input file contains no valid queries.")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Load documents from the database
    docs = load_documents()

    # Create the retrieval engine instance and fit it with the documents
    engine = RetrievalEngine(
        use_prf=False,
        use_rerank=True,
    )
    engine.fit(corpus=docs)
    print(f"Found {len(docs)} documents in the database.")

    # Run the queries and save the results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for i, query in enumerate(queries, 1):
            try:
                results = engine.search(query=query)
                out.write(f"Query {i}: {query}\n")
                for j, doc in enumerate(results, 1):
                    out.write(f"  {j}. {doc.title or '(no title)'} — {doc.url}\n")
                out.write("\n")
            except Exception as e:
                out.write(f"Query {i}: {query}\n")
                out.write(f"  Error: {str(e)}\n\n")
                print(f"Error processing query {i}: {e}")

    print(f"Saved results to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    run_batch_queries()
