import os
import gzip
import duckdb
from typing import List
from retrieval_engine.core.engine import RetrievalEngine
from retrieval_engine.docs.document_store import Document

INPUT_FILE = "queries/queries.txt"
OUTPUT_FILE = "results/results.txt"
DB_PATH = "../crawler/crawler_2/final/data.db"

def load_documents() -> List[Document]:
    con = duckdb.connect(DB_PATH)
    rows = con.execute("SELECT link, title, content, image_url FROM main.documents_filtered3").fetchall()

    documents = []
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
            favicon=None  # Favicon ist im aktuellen Schema nicht vorhanden
        )
        documents.append(doc)
    return documents

def run_batch_queries():
    if not os.path.exists(INPUT_FILE):
        print(f"Fehler: '{INPUT_FILE}' nicht gefunden.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    if not queries:
        print("Keine gültigen Queries gefunden.")
        return

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    docs = load_documents()

    engine = RetrievalEngine(use_prf=False, use_rerank=True)
    engine.fit(docs)
    print(f"→ {len(docs)} Dokumente geladen.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for i, query in enumerate(queries, 1):
            try:
                results = engine.search(query)
                out.write(f"Query {i}: {query}\n")
                for j, doc in enumerate(results, 1):
                    out.write(f"  {j}. {doc.title or '(kein Titel)'} — {doc.url}\n")
                out.write("\n")
            except Exception as e:
                out.write(f"Query {i} ERROR: {e}\n\n")

    print(f"Ergebnisse gespeichert in: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_batch_queries()
