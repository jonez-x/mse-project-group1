from datetime import datetime
import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from retrieval_engine.core.engine import RetrievalEngine
from retrieval_engine.docs.document_store import DocumentStore, Document
import duckdb
from contextlib import asynccontextmanager


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


document_store: Optional[DocumentStore] = None
retrieval_engine: Optional[RetrievalEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global document_store, retrieval_engine
    logger.info("Loading documents from DuckDB...")

    document_store = DocumentStore()

    try:
        con = duckdb.connect("crawler/tuebingen_crawl.duckdb")
        rows = con.execute("SELECT url, title, excerpt, main_image, favicon FROM main.crawl_results").fetchall()

        for row in rows:
            doc = Document(*row)
            document_store.add_document(doc)

        logger.info(f"Loaded {len(document_store.get_all())} documents.")

        retrieval_engine = RetrievalEngine(use_prf=False, use_rerank=False)
        retrieval_engine.fit(document_store.get_all())
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="Tübingen Search API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware - allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class SearchResponse(BaseModel):
    results: List[str]


class CrawlRequest(BaseModel):
    urls: List[str]
    max_pages: Optional[int] = 1000


class BatchSearchRequest(BaseModel):
    queries: List[str]


# Placeholder functions that need to be implemented
def crawl(frontier, index):
    # TODO: Implement crawler functionality
    pass


def index(doc, index_path):
    # TODO: Implement indexing functionality
    pass


def retrieve(query, index_path):
    # TODO: Implement retrieval logic
    # Return mock data for now
    return [
        f"Result 1 for '{query}'",
        f"Result 2 for '{query}'",
        f"Result 3 for '{query}'"
    ]


def batch(results):
    # TODO: Implement batch processing
    pass




@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Tübingen Search API",
        "version": "1.0.0",
        "description": "Web crawling and search system for Tübingen content",
        "endpoints": {
            "search": "/search?q=query",
            "crawl": "/crawl",
            "batch_search": "/batch-search",
            "health": "/health"
        }
    }


@app.get("/search", response_model=SearchResponse)
async def search(q: str = Query(..., min_length=1, description="Search query")) -> SearchResponse:
    """
    Method to search for documents (GET request) using a query string.
    
    :param q: Search query string
    :returns: A SearchResponse object containing the results
    """
    try:
        logger.info(f"Received search query: {q}")

        # TODO: Replace with actual retrieval logic when crawler/retriever is implemented
        # For now, return simple test results that match frontend expectations

        # Simple test logic for frontend testing - REMOVE WHEN IMPLEMENTING REAL SEARCH
        if q == "result1":
            return SearchResponse(results=["result1"])
        elif q == "result2":
            return SearchResponse(results=["result2"])
        elif q == "result3":
            return SearchResponse(results=["result3"])

        # Default mock results
        results = retrieve(q, index_path="data/index")
        return SearchResponse(results=results)

    except Exception as e:
        logger.error(f"Error processing search query '{q}': {str(e)}")
        raise HTTPException(status_code=500, detail="Search failed")


@app.post("/crawl")
async def start_crawl(request: CrawlRequest) -> Dict[str, Any]:
    """
    Start a web crawl with the provided seed URLs and options (POST request).

    :param request: A CrawlRequest object containing URLs and options
    :returns: Status message indicating crawl has started
    """
    try:
        logger.info(f"Starting crawl with {len(request.urls)} seed URLs")

        # TODO: Implement actual crawling
        # For now, return success message
        frontier = {
            "urls": request.urls,
            "max_pages": request.max_pages
        }

        # crawl(frontier, "data/index")  # Uncomment once it is implemented

        return {
            "message": "Crawling started",
            "seed_urls": request.urls,
            "max_pages": request.max_pages,
            "status": "started"
        }

    except Exception as e:
        logger.error(f"Error starting crawl: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start crawling")


@app.post("/batch-search")
async def batch_search(request: BatchSearchRequest):
    """
    Method to perform batch search on multiple queries at once (POST request).

    :param request: A BatchSearchRequest object containing list of queries
    :returns: A dictionary with batch search results
    """
    try:
        logger.info(f"Processing batch search with {len(request.queries)} queries")

        # TODO: Implement actual batch processing
        results = {}
        for i, query in enumerate(request.queries, 1):
            results[f"query_{i}"] = retrieve(query, "data/index")

        return {
            "message": "Batch search completed",
            "num_queries": len(request.queries),
            "results": results
        }

    except Exception as e:
        logger.error(f"Error in batch search: {str(e)}")
        raise HTTPException(status_code=500, detail="Batch search failed")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": str(datetime.now()),
        "version": "1.0.0",
        "message": "API is running",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
