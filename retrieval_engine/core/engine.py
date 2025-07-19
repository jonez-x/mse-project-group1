from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union, Optional

from retrieval_engine.enhancement import CrossEncoderReRanker, RocchioPRF
from retrieval_engine.fusion import ReciprocalRankFusion
from retrieval_engine.retrievers import BM25Retriever, DenseRetriever


class RetrievalEngine:
    """
    Implements a retrieval engine that combines sparse and dense retrieval methods
    with optional pseudo-relevance feedback (PRF) and re-ranking for improved
    search results.

    The class integrates multiple retrieval submodules: BM25Retriever for sparse
    retrieval, DenseRetriever for dense retrieval, Reciprocal Rank Fusion (RRF)
    for combining retrieval scores, optionally RocchioPRF for query refinement and
    CrossEncoderReRanker for re-ranking results. It is designed to perform a multi-step
    search process where each component can be tailored as needed.

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
            bm25_params: Union[Dict[str, Union[float, int]], None] = None,
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
            bm25_params (Dict[str, Union[float, int]] | None): Parameters for BM25 retriever.
            dense_model_name (str): The name of the dense retriever model (default: "all-MiniLM-L6-v2").
            rrf_k (int): The top k results to consider in Reciprocal Rank Fusion (default: 60).
            use_prf (bool): Whether to apply pseudo-relevance feedback (PRF) to the search results (default: False).
            prf_params (Dict[str, float] | None): Optional parameters for initializing the PRF module.
            use_rerank (bool): Whether to apply re-ranking to the final results (default: False).
            rerank_params (Dict[str, Union[str, int, bool]] | None): Optional parameters for initializing the re-ranker.

        """
        # Initialize the retrieval components
        self.bm25 = BM25Retriever(**(bm25_params or {}))
        self.dense = DenseRetriever(model_name=dense_model_name)
        self.rrf = ReciprocalRankFusion(k=rrf_k)

        # When using PRF, initialize the RocchioPRF module
        self.use_prf = use_prf
        self.prf = RocchioPRF(**(prf_params or {})) if use_prf else None

        # When using re-ranking, initialize the CrossEncoderReRanker module
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
        # Query both BM25 and dense retrievers
        dense_hits = self.dense.query(
            query,
            top_k=dense_top_k
        )
        bm25_ids, _ = self.bm25.query(
            query,
            top_k=bm25_top_k
        )

        #
        dense_ids = [str(doc_id) for doc_id, _, _ in dense_hits]
        bm25_ids_str = [str(doc_id) for doc_id in bm25_ids]

        fused = self.rrf.fuse(
            [bm25_ids_str, dense_ids],
            top_k=max(final_top_k, bm25_top_k, dense_top_k),
            return_scores=True,
        )

        if self.use_prf and fused:
            top_doc_ids = [doc_id for doc_id, _ in fused[:10]]
            top_doc_texts = [doc for doc_id, doc in self.bm25.get_docs([int(did) for did in top_doc_ids])]
            rel_vectors = self.dense.embed_documents(top_doc_texts)
            query_vec = self.dense.embed_query(query)
            refined_vec = self.prf.refine(query_vec, rel_vectors)

            dense_hits_refined = self.dense.search_from_vector(refined_vec, top_k=dense_top_k)
            dense_ids_refined = [str(doc_id) for doc_id, _, _ in dense_hits_refined]

            fused = self.rrf.fuse(
                [bm25_ids_str, dense_ids_refined],
                top_k=max(final_top_k, bm25_top_k, dense_top_k),
                return_scores=True,
            )

        if self.use_rerank and fused:
            doc_ids = [int(doc_id) for doc_id, _ in fused[:final_top_k]]
            docs_text = self.bm25.get_docs(doc_ids)
            reranked = self.reranker.rerank(query, docs_text, top_n=final_top_k)
            fused_dict = dict(fused)
            final_hits = [(doc_id, fused_dict.get(doc_id, 0.0)) for doc_id, _ in reranked]
        else:
            final_hits = fused[:final_top_k]

        return final_hits
