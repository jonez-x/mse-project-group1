from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

from retrieval_engine.enhancement import CrossEncoderReRanker, RocchioPRF
from retrieval_engine.fusion import ReciprocalRankFusion
from retrieval_engine.retrievers import BM25Retriever, DenseRetriever


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
        # Initialize the core retrieval components
        self.bm25 = BM25Retriever(**(bm25_params or {}))
        self.dense = DenseRetriever(model_name=dense_model_name)
        self.rrf = ReciprocalRankFusion(k=rrf_k)

        # Initialize pseudo-relevance feedback if enabled
        self.use_prf = use_prf
        self.prf = RocchioPRF(**(prf_params or {})) if use_prf else None

        # Initialize re-ranking if enabled
        self.use_rerank = use_rerank
        self.reranker = (
            CrossEncoderReRanker(**(rerank_params or {})) if use_rerank else None
        )

    def fit(self, corpus: Sequence[str]) -> None:
        """
        Build the index for the given corpus and fit the retrievers.

        Parameters:
            corpus (Sequence[str]): A sequence of documents to index. Each document is a string.
        """
        # Convert the corpus to a list for consistent indexing
        corpus_list = list(corpus)

        # Fit the BM25 and dense retrievers with the corpus
        self.bm25.fit(docs=corpus)
        self.dense.fit(corpus=corpus_list)

    def search(
            self,
            query: str,
            bm25_top_k: int = 300,
            dense_top_k: int = 300,
            final_top_k: int = 100,
    ) -> List[Tuple[str, float]]:
        """
        Execute the complete retrieval pipeline and return top documents with their scores.

        Parameters:
            query (str): The search query string.
            bm25_top_k (int): The number of top documents to retrieve using BM25 (default: 300).
            dense_top_k (int): The number of top documents to retrieve using dense retrieval (default: 300).
            final_top_k (int): The number of final top documents to return after fusion and re-ranking (default: 100).

        Returns:
            List[Tuple[str, float]]: A list of tuples containing document IDs and their scores,
                                     sorted by relevance to the query.
        """
        # Step 1: Query both BM25 and dense retrievers independently
        dense_hits = self.dense.query(
            query=query,
            top_k=dense_top_k
        )
        bm25_ids, _ = self.bm25.query(
            text=query,
            top_k=bm25_top_k
        )

        # Convert document IDs to strings for RRF fusion
        dense_ids = [str(doc_id) for doc_id, _, _ in dense_hits]
        bm25_ids_str = [str(doc_id) for doc_id in bm25_ids]

        # Step 2: Fuse the results using Reciprocal Rank Fusion
        fused = self.rrf.fuse(
            [bm25_ids_str, dense_ids],
            top_k=max(final_top_k, bm25_top_k, dense_top_k),
            return_scores=True,
        )

        # Step 3: Apply pseudo-relevance feedback if enabled
        if self.use_prf and fused:
            # Use top 10 documents as relevant feedback for query expansion
            top_doc_ids = [doc_id for doc_id, _ in fused[:10]]
            top_doc_texts = [doc for doc_id, doc in self.bm25.get_docs([int(did) for did in top_doc_ids])]

            # Encode the relevant documents and refine the query vector
            rel_vectors = self.dense.embed_documents(docs=top_doc_texts)
            query_vec = self.dense.embed_query(query=query)
            refined_vec = self.prf.refine(
                query_vec=query_vec,
                rel_doc_vecs=rel_vectors
            )

            # Re-run dense retrieval with the refined query vector
            dense_hits_refined = self.dense.search_from_vector(
                query_vec=refined_vec,
                top_k=dense_top_k
            )
            dense_ids_refined = [str(doc_id) for doc_id, _, _ in dense_hits_refined]

            # Re-fuse with the refined dense results
            fused = self.rrf.fuse(
                [bm25_ids_str, dense_ids_refined],
                top_k=max(final_top_k, bm25_top_k, dense_top_k),
                return_scores=True,
            )

        # Step 4: Apply cross-encoder re-ranking if enabled
        if self.use_rerank and fused:
            # Get the top candidates for re-ranking
            doc_ids = [int(doc_id) for doc_id, _ in fused[:final_top_k]]
            docs_text = self.bm25.get_docs(doc_ids=doc_ids)

            # Re-rank using cross-encoder model
            reranked = self.reranker.rerank(
                query=query,
                doc_pairs=docs_text,
                top_n=final_top_k,
            )

            # Preserve original fusion scores for documents that were re-ranked
            fused_dict = dict(fused)
            final_hits = [(doc_id, fused_dict.get(doc_id, 0.0)) for doc_id, _ in reranked]
        else:
            # Use fusion results directly if no re-ranking
            final_hits = fused[:final_top_k]

        return final_hits
