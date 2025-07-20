import gzip
import logging
from fastapi import FastAPI, HTTPException, Query, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from nltk import download
from pydantic import BaseModel
from typing import List, Optional, Dict
import duckdb
from retrieval_engine.core.engine import RetrievalEngine
from retrieval_engine.docs.document_store import Document
from contextlib import asynccontextmanager
import uvicorn
from collections import Counter
from nltk.corpus import stopwords
import re

download("stopwords")
stop_words = set(stopwords.words("english"))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

retriever_v1: Optional[RetrievalEngine] = None
retriever_v2: Optional[RetrievalEngine] = None


def compute_tf(text: str) -> Dict[str, int]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return dict(Counter(tokens))

def load_documents_v1() -> List[Document]:
    con = duckdb.connect("crawler/tuebingen_crawl.duckdb")
    rows = con.execute("SELECT url, title, excerpt, main_image, favicon FROM main.crawl_results").fetchall()

    documents = []
    for url, title, excerpt, img, favicon in rows:
        tf = compute_tf((title or "") + " " + (excerpt or ""))
        doc = Document(url, title, excerpt, img, favicon)
        doc.word_dict = tf
        documents.append(doc)
    return documents



def load_documents_v2() -> List[Document]:
    con = duckdb.connect("crawler/crawler_2/data.db")
    rows = con.execute("SELECT link, title, content FROM main.documents").fetchall()

    documents = []
    for link, title, blob in rows:
        try:
            content = gzip.decompress(blob).decode("utf-8", errors="ignore")
        except Exception:
            content = ""

        excerpt = content[:300]
        tf = compute_tf((title or "") + " " + content)
        doc = Document(
            url=link,
            title=title,
            excerpt=excerpt,
            main_image=None,
            favicon=None
        )
        doc.word_dict = tf
        documents.append(doc)
    return documents



@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever_v1, retriever_v2

    # v1: Tübingen Crawler
    docs1 = load_documents_v1()
    retriever_v1 = RetrievalEngine(use_prf=True, use_rerank=True)
    retriever_v1.fit(docs1)
    logger.info(f"Loaded {len(docs1)} documents from v1.")

    # v2: Gzipped Content
    docs2 = load_documents_v2()
    retriever_v2 = RetrievalEngine(use_prf=True, use_rerank=True)
    retriever_v2.fit(docs2)
    logger.info(f"Loaded {len(docs2)} documents from v2.")

    yield


app = FastAPI(title="Tübingen Search API", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Doc(BaseModel):
    id: int
    url: str
    favicon: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    image: Optional[str] = None
    word_dictionary: Optional[Dict[str, float]] = None,
    document_length: Optional[int]



class SearchResponse(BaseModel):
    results: List[Doc]


class BatchSearchRequest(BaseModel):
    queries: List[str]


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())

def build_docs(docs: List[Document], query: str) -> List[Doc]:
    print(query)
    query_words = set(tokenize(query)) - stop_words
    print(stop_words)
    print(query_words)
    return [
        Doc(
            id=i + 1,
            url=doc.url,
            title=doc.title,
            description=doc.excerpt,
            favicon=doc.favicon,
            image=doc.main_image,
            word_dictionary={
                word: count
                for word, count in getattr(doc, "word_dict", {}).items()
                if word.lower() in query_words
            },
            document_length= len(re.findall(r"\b\w+\b", doc.excerpt.lower()))
        )
        for i, doc in enumerate(docs)
    ]



# --- API Router für v1 ---
router_v1 = APIRouter(prefix="/v1", tags=["v1"])

@router_v1.get("/search", response_model=SearchResponse)
async def search_v1(q: str = Query(...)):
    print()
    results = retriever_v1.search(q)
    return SearchResponse(results=build_docs(results, q))


# --- API Router für v2 ---
router_v2 = APIRouter(prefix="/v2", tags=["v2"])

@router_v2.get("/search", response_model=SearchResponse)
async def search_v2(q: str = Query(...)):
    results = retriever_v2.search(q)
    return SearchResponse(results=build_docs(results, q))


app.include_router(router_v1)
app.include_router(router_v2)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
