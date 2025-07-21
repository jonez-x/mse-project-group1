from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple, Union
from nltk import download
from nltk.corpus import stopwords
from retrieval_engine.enhancement import CrossEncoderReRanker, RocchioPRF
from retrieval_engine.fusion import ReciprocalRankFusion
from retrieval_engine.retrievers import BM25Retriever, DenseRetriever
from retrieval_engine.docs.document_store import Document, DocumentStore

# Download NLTK stopwords list if not already present
download("stopwords")
stop_words = set(stopwords.words("english"))

class RetrievalEngine:
    """
    Implements a retrieval engine that combines sparse (BM25) and dense
    (Sentence-BERT) retrieval methods, with optional pseudo-relevance feedback
    and cross-encoder re-ranking. During ingestion, near-duplicate titles
    (>= 99% similarity) are filtered to speed up indexing and reduce noise.
    """

    def __init__(
            self,
            bm25_params: Optional[Dict[str, Union[float, int]]] = None,
            dense_model_name: str = "all-MiniLM-L6-v2",
            rrf_k: int = 60,
            use_prf: bool = False,
            prf_params: Optional[Dict[str, float]] = None,
            use_rerank: bool = False,
            rerank_params: Optional[Dict[str, Union[str, int, bool]]] = None,
    ) -> None:
        """
        Initialize the RetrievalEngine.

        Args:
            bm25_params: Parameters for the BM25 retriever (e.g., k1, b).
            dense_model_name: SentenceTransformer model name for dense retrieval.
            rrf_k: Parameter controlling decay in Reciprocal Rank Fusion.
            use_prf: If True, apply Rocchio pseudo-relevance feedback.
            prf_params: Parameters for the Rocchio PRF module.
            use_rerank: If True, apply cross-encoder re-ranking.
            rerank_params: Parameters for the CrossEncoderReRanker.
        """
        # Document store holds all ingested documents
        self.store = DocumentStore()
        # Sparse retriever
        self.bm25 = BM25Retriever(**(bm25_params or {}))
        # Dense retriever (embedding-based)
        self.dense = DenseRetriever(model_name=dense_model_name, device=None)
        # Fusion module to combine rankings
        self.rrf = ReciprocalRankFusion(k=rrf_k)

        # Optional pseudo-relevance feedback
        self.use_prf = use_prf
        self.prf = RocchioPRF(**(prf_params or {})) if use_prf else None
        # Optional cross-encoder re-ranking
        self.use_rerank = use_rerank
        self.reranker = (
            CrossEncoderReRanker(**(rerank_params or {})) if use_rerank else None
        )

    def fit(self, corpus: Sequence[Document]) -> None:
        """
        Build the index for the given corpus and fit both BM25 and dense retrievers.

        During ingestion, documents with nearly identical titles (>=99% similarity)
        are skipped (only the first occurrence is kept), using a simple bucketing
        by title prefix to reduce comparisons.

        Args:
            corpus: Sequence of Document objects to index.
        """
        buckets: Dict[str, List[str]] = {}
        deduped_docs: List[Document] = []

        for doc in corpus:
            title_norm = doc.title.strip().lower()
            # Bucket key based on first 8 characters of the normalized title
            bucket_key = title_norm[:8]
            seen = buckets.setdefault(bucket_key, [])
            # If a very similar title is already seen in this bucket, skip document
            if any(SequenceMatcher(None, title_norm, prev).ratio() >= 0.99 for prev in seen):
                continue
            seen.append(title_norm)

            # Preprocess the document text to generate an excerpt without stopwords
            text = doc.to_text()
            tokens = self.tokenize(text)
            filtered = [t for t in tokens if t not in stop_words]
            doc.excerpt = " ".join(filtered)

            # Add to the in-memory store and deduplicated list
            self.store.add_document(doc)
            deduped_docs.append(doc)

        # Fit retrievers on the deduplicated corpus
        self.bm25.fit(docs=deduped_docs)
        self.dense.fit(corpus=deduped_docs)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize input text into lowercase word tokens.
        """
        return re.findall(r"\b\w+\b", text.lower())

    def search(
            self,
            query: str,
            bm25_top_k: int = 300,
            dense_top_k: int = 300,
            final_top_k: int = 100,
    ) -> List[Union[Document, Tuple[Document, float]]]:
        """
        Execute the full retrieval pipeline:
          1. Sparse (BM25) and dense retrieval
          2. Reciprocal Rank Fusion (RRF)
          3. Optional pseudo-relevance feedback (Rocchio)
          4. Optional cross-encoder re-ranking

        Args:
            query: User query string.
            bm25_top_k: Number of BM25 candidates.
            dense_top_k: Number of dense candidates.
            final_top_k: Number of final results to return.

        Returns:
            List of Documents or (Document, score) tuples, depending on reranking.
        """
        # Clean and filter query
        tokens = self.tokenize(query)
        filtered_tokens = [t for t in tokens if t not in stop_words]
        clean_query = " ".join(filtered_tokens)

        # Step 1: retrieve candidates
        dense_hits = self.dense.query(query=clean_query, top_k=dense_top_k)
        bm25_hits = self.bm25.query(query=clean_query, top_k=bm25_top_k)
        dense_urls = [doc.url for _, _, doc in dense_hits]
        bm25_urls = [doc.url for doc in bm25_hits]

        # Step 2: fuse rankings
        fused = self.rrf.fuse(
            [bm25_urls, dense_urls],
            top_k=max(bm25_top_k, dense_top_k, final_top_k),
            return_scores=True,
        )

        # Step 3: pseudo-relevance feedback (optional)
        if self.use_prf and fused:
            top_urls = [url for url, _ in fused[:10]]
            top_docs = self.store.get_by_ids(top_urls)
            rel_vecs = self.dense.embed_documents([d.to_text() for d in top_docs])
            query_vec = self.dense.embed_query(clean_query)
            refined_vec = self.prf.refine(query_vec, rel_doc_vecs=rel_vecs)
            dense_hits = self.dense.search_from_vector(refined_vec, top_k=dense_top_k)
            dense_urls = [d.url for _, _, d in dense_hits]
            fused = self.rrf.fuse([bm25_urls, dense_urls], top_k=len(fused), return_scores=True)

        # Step 4: collect final documents
        final_ids = [url for url, _ in fused[:final_top_k]]
        final_docs = self.store.get_by_ids(final_ids)

        # Optional re-ranking
        if self.use_rerank:
            reranked = self.reranker.rerank(query=clean_query, doc_pairs=final_docs, top_n=final_top_k)
            return [(self.store.get_by_ids([doc_id])[0], score) for doc_id, score in reranked]

        return final_docs
