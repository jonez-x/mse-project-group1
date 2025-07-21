import argparse
import gzip
import logging
import re
from collections import Counter
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import duckdb
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from nltk import download
from nltk.corpus import stopwords
from pydantic import BaseModel

from autocomplete_system.services.autocomplete import AutocompleteService, ModelType
from config import DEFAULT_AUTOCOMPLETE_MODEL
from retrieval_engine.core.engine import RetrievalEngine
from retrieval_engine.docs.document_store import Document

# Global variable to store the selected model
SELECTED_AUTOCOMPLETE_MODEL = DEFAULT_AUTOCOMPLETE_MODEL

download("stopwords")
stop_words = set(stopwords.words("english"))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

retriever_v1: Optional[RetrievalEngine] = None
retriever_v2: Optional[RetrievalEngine] = None
autocomplete_service: Optional[AutocompleteService] = None


def compute_tf(text: str) -> Dict[str, int]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return dict(Counter(tokens))

def load_documents() -> List[Document]:
    con = duckdb.connect("crawler/crawler_2/final/final.db")
    rows = con.execute("SELECT link, title, content, image_url FROM main.documents").fetchall()

    documents = []
    for link, title, content, image_url in rows:
        try:
            content = gzip.decompress(content).decode("utf-8", errors="ignore")
        except Exception:
            content = ""

        excerpt = content[:300]
        tf = compute_tf((title or "") + " " + content)
        doc = Document(
            url=link,
            title=title,
            excerpt=excerpt,
            main_image=image_url,
            favicon=None
        )
        doc.word_dict = tf
        documents.append(doc)
    return documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever_v1, retriever_v2, autocomplete_service

    # v1: Tübingen Crawler
    docs = load_documents()
    retriever_v1 = RetrievalEngine(use_prf=True, use_rerank=True)
    retriever_v1.fit(docs)
    logger.info(f"Loaded {len(docs)} documents from v1.")

    # Initialize autocomplete service
    requested_model = ModelType.NGRAM if SELECTED_AUTOCOMPLETE_MODEL == "ngram" else ModelType.DATAMUSE
    logger.info(f"Initializing autocomplete service with requested model: {SELECTED_AUTOCOMPLETE_MODEL}")

    autocomplete_service = AutocompleteService(default_model=requested_model)

    available_models = autocomplete_service.get_available_models()
    logger.info(f"Autocomplete service initialized. Available models: {available_models}")

    # Validate that the requested model is actually available
    if SELECTED_AUTOCOMPLETE_MODEL not in available_models:
        logger.warning(f"Requested model '{SELECTED_AUTOCOMPLETE_MODEL}' is not available!")
        logger.warning(f"Will fallback to available models: {available_models}")
        if not available_models:
            logger.error("No autocomplete models are available!")
        else:
            logger.info(f"Service will use: {available_models[0]} as fallback")
    else:
        logger.info(f"✓ Successfully loaded requested model: {SELECTED_AUTOCOMPLETE_MODEL}")

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


class AutocompleteSuggestion(BaseModel):
    word: str
    score: Optional[float]
    type: str  # "completion" or "next_word"
    model: str
    full_query: str


class AutocompleteResponse(BaseModel):
    suggestions: List[AutocompleteSuggestion]
    model_used: str
    query: str
    count: int


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def build_docs(docs: List[Document], query: str) -> List[Doc]:
    query_words = set(tokenize(query)) - stop_words
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
            document_length=len(re.findall(r"\b\w+\b", doc.excerpt.lower()))
        )
        for i, doc in enumerate(docs)
    ]


# --- API Router für v1 ---
router_v1 = APIRouter(prefix="/v1", tags=["v1"])

@router_v1.get("/search", response_model=SearchResponse)
async def search_v1(q: str = Query(...)):
    results = retriever_v1.search(q)
    return SearchResponse(results=build_docs(results, q))

app.include_router(router_v1)


# --- Autocomplete API ---
@app.get("/autocomplete", response_model=AutocompleteResponse)
async def get_autocomplete(
        q: str = Query(..., description="Query text for autocomplete"),
        model: str = Query(SELECTED_AUTOCOMPLETE_MODEL, description="Model to use: 'ngram' or 'datamuse'"),
        max_suggestions: int = Query(5, description="Maximum number of suggestions", ge=1, le=10)
):
    """Get autocomplete suggestions for a query."""
    if not autocomplete_service:
        raise HTTPException(status_code=500, detail="Autocomplete service not initialized")

    # Parse model type
    try:
        model_type = ModelType(model.lower())
    except ValueError:
        raise HTTPException(status_code=400,
                            detail=f"Invalid model. Available: {autocomplete_service.get_available_models()}")

    # Check if model is available
    if model_type.value not in autocomplete_service.get_available_models():
        raise HTTPException(status_code=400,
                            detail=f"Model '{model}' not available. Available: {autocomplete_service.get_available_models()}")

    try:
        suggestions_data = autocomplete_service.get_suggestions(q, model_type, max_suggestions)

        suggestions = [
            AutocompleteSuggestion(
                word=s["word"],
                score=s["score"],
                type=s["type"],
                model=s["model"],
                full_query=s["full_query"]
            )
            for s in suggestions_data
        ]

        return AutocompleteResponse(
            suggestions=suggestions,
            model_used=model_type.value,
            query=q,
            count=len(suggestions)
        )

    except Exception as e:
        logger.error(f"Autocomplete error: {e}")
        raise HTTPException(status_code=500, detail=f"Autocomplete failed: {str(e)}")


@app.get("/autocomplete/models")
async def get_available_models():
    """Get list of available autocomplete models."""
    if not autocomplete_service:
        raise HTTPException(status_code=500, detail="Autocomplete service not initialized")

    return {
        "available_models": autocomplete_service.get_available_models(),
        "default_model": autocomplete_service.default_model.value
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Tübingen Search API")
    parser.add_argument(
        "--model",
        choices=["ngram", "datamuse"],
        default=DEFAULT_AUTOCOMPLETE_MODEL,
        help=f"Autocomplete model to use (default: {DEFAULT_AUTOCOMPLETE_MODEL})"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Set the selected model globally
    SELECTED_AUTOCOMPLETE_MODEL = args.model

    logger.info(f"Starting server with autocomplete model: {SELECTED_AUTOCOMPLETE_MODEL}")

    # Start the Backend server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )
