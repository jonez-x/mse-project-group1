from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple, Union
from nltk import download
from nltk.corpus import stopwords
from retrieval_engine.enhancement import CrossEncoderReRanker, RocchioPRF
from retrieval_engine.fusion import ReciprocalRankFusion
from retrieval_engine.retrievers import BM25Retriever, DenseRetriever
from retrieval_engine.docs.document_store import Document, DocumentStore

download("stopwords")
stop_words = set(stopwords.words("english"))

class RetrievalEngine:
    """
    Implements a retrieval engine that combines sparse and dense retrieval methods
    with optional pseudo-relevance feedback (PRF) and re-ranking for improved
    search results.

    Attributes:
        bm25 (BM25Retriever): The BM25 retriever for sparse retrieval.
        dense (DenseRetriever): The dense retriever for embedding-based retrieval.
        rrf (ReciprocalRankFusion): The RRF module for combining results from different retrievers.
        use_prf (bool): Flag indicating whether to use pseudo-relevance feedback.
        prf (RocchioPRF | None): The PRF module for refining queries, if used.
        use_rerank (bool): Flag indicating whether to use re-ranking.
        reranker (CrossEncoderReRanker | None): The re-ranker for final result adjustment, if used.
        store (DocumentStore): Stores and manages all documents in memory.
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
        Initialize the retrieval engine with specified parameters.

        Parameters:
            bm25_params: Optional parameters for BM25 retriever (k1, b, use_numpy).
                If None, uses default BM25 parameters.
            dense_model_name: The name of the sentence transformer model for dense retrieval.
                Defaults to "all-MiniLM-L6-v2", a lightweight and efficient model.
            rrf_k: The k parameter for Reciprocal Rank Fusion, controlling how quickly
                the reciprocal term decays (default: 60).
            use_prf: Whether to apply Rocchio pseudo-relevance feedback for query expansion.
                When enabled, uses top retrieved documents to refine the query (default: False).
            prf_params: Optional parameters for the Rocchio PRF module (alpha, beta, gamma).
                If None, uses default Rocchio parameters.
            use_rerank: Whether to apply cross-encoder re-ranking to final results.
                Improves precision but increases computational cost (default: False).
            rerank_params: Optional parameters for the cross-encoder re-ranker
                (model_name, batch_size, device, normalize). If None, uses defaults.
        """
        self.store = DocumentStore()
        self.bm25 = BM25Retriever(**(bm25_params or {}))
        self.dense = DenseRetriever(model_name=dense_model_name, device=None)   # use best available device
        self.rrf = ReciprocalRankFusion(k=rrf_k)

        # Initialize pseudo-relevance feedback if enabled
        self.use_prf = use_prf
        self.prf = RocchioPRF(**(prf_params or {})) if use_prf else None

        # Initialize re-ranking if enabled
        self.use_rerank = use_rerank
        self.reranker = (
            CrossEncoderReRanker(**(rerank_params or {})) if use_rerank else None
        )

    def fit(self, corpus: Sequence[Document]) -> None:
        """
        Build the index for the given corpus and fit the retrievers.

        Parameters:
            corpus (Sequence[Document]): A sequence of Document objects to index.
        """
        for doc in corpus:
            text = doc.to_text()
            tokens = self.tokenize(text)
            filtered_tokens = [t for t in tokens if t not in stop_words]
            excerpt = " ".join(filtered_tokens)
            doc.excerpt = excerpt
            self.store.add_document(doc)

        self.bm25.fit(docs=corpus)
        self.dense.fit(corpus=corpus)

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def search(
            self,
            query: str,
            bm25_top_k: int = 300,
            dense_top_k: int = 300,
            final_top_k: int = 100,
    ) -> List[Document]:
        """
        Execute the complete retrieval pipeline and return top documents with their scores.

        Parameters:
            query (str): The search query string.
            bm25_top_k (int): The number of top documents to retrieve using BM25 (default: 300).
            dense_top_k (int): The number of top documents to retrieve using dense retrieval (default: 300).
            final_top_k (int): The number of final top documents to return after fusion and re-ranking (default: 100).

        Returns:
            List[Tuple[Document, float]]: A list of tuples containing Document objects and their scores,
                                          sorted by relevance to the query.
        """

        tokens = self.tokenize(query)
        filtered_tokens = [token for token in tokens if token not in stop_words]
        query = " ".join(filtered_tokens)
        print(query)

        # Step 1: Get candidates from both retrievers
        dense_hits = self.dense.query(query=query, top_k=dense_top_k)
        bm25_hits = self.bm25.query(query=query, top_k=bm25_top_k)
        # Extract doc URLs for fusion
        dense_urls = [doc.url for _, _, doc in dense_hits]
        bm25_urls = [doc.url for doc in bm25_hits]

        # Step 2: RRF fusion
        fused = self.rrf.fuse(
            [bm25_urls, dense_urls],
            top_k=max(final_top_k, bm25_top_k, dense_top_k),
            return_scores=True,
        )

        # Step 3: Optional PRF (Pseudo-Relevance Feedback)
        if self.use_prf and fused:
            top_urls = [url for url, _ in fused[:10]]
            top_docs = self.store.get_by_ids(top_urls)
            rel_vecs = self.dense.embed_documents([doc.to_text() for doc in top_docs])
            query_vec = self.dense.embed_query(query)
            refined_vec = self.prf.refine(query_vec, rel_doc_vecs=rel_vecs)
            dense_hits_refined = self.dense.search_from_vector(refined_vec, top_k=dense_top_k)
            dense_urls_refined = [doc.url for _, _, doc in dense_hits_refined]

            fused = self.rrf.fuse(
                [bm25_urls, dense_urls_refined],
                top_k=max(final_top_k, bm25_top_k, dense_top_k),
                return_scores=True,
            )

        # Step 4: Re-ranking or plain scoring
        final_urls = [url for url, _ in fused[:final_top_k]]
        final_docs = self.store.get_by_ids(final_urls)

        if self.use_rerank:
            reranked = self.reranker.rerank(query=query, doc_pairs=final_docs, top_n=final_top_k)
            final_hits = [
                (self.store.get_by_ids([doc_id])[0], score)
                for doc_id, score in reranked
            ]
        else:
            final_hits = [
                doc
                for doc in final_docs
            ]

        return final_hits
