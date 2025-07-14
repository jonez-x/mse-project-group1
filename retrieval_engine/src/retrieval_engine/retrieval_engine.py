from __future__ import annotations
from typing import List, Sequence, Tuple
from bm25_retriever import BM25Retriever
from dense_retriever import DenseRetriever
from rrf import ReciprocalRankFusion
from rocchio_prf import RocchioPRF
from cross_encoder_reranker import CrossEncoderReRanker


class RetrievalEngine:
    """End‑to‑End‑Pipeline, die Sparse + Dense Retrieval kombiniert, per RRF
    fusioniert, optional PRF anwendet und schließlich (ebenfalls optional) per
    Cross‑Encoder re‑rankt.
    """

    def __init__(
        self,
        bm25_params: dict | None = None,
        dense_model_name: str = "all-MiniLM-L6-v2",
        rrf_k: int = 60,
        use_prf: bool = False,
        prf_params: dict | None = None,
        use_rerank: bool = False,
        rerank_params: dict | None = None,
    ) -> None:
        self.bm25 = BM25Retriever(**(bm25_params or {}))
        self.dense = DenseRetriever(model_name=dense_model_name)
        self.rrf = ReciprocalRankFusion(k=rrf_k)

        self.use_prf = use_prf
        self.prf = RocchioPRF(**(prf_params or {})) if use_prf else None

        self.use_rerank = use_rerank
        self.reranker = (
            CrossEncoderReRanker(**(rerank_params or {})) if use_rerank else None
        )

    def fit(self, corpus: Sequence[str]) -> None:
        """Baue die Indizes für BM25 und Dense Retriever auf."""
        corpus_list = list(corpus)
        self.bm25.fit(corpus)
        self.dense.fit(corpus_list)

    def search(
        self,
        query: str,
        bm25_top_k: int = 300,
        dense_top_k: int = 300,
        final_top_k: int = 100,
    ) -> List[Tuple[str, float]]:
        """Führt die komplette Pipeline aus und liefert Top‑Dokumente mit Score."""

        bm25_hits = self.bm25.query(query, top_k=bm25_top_k)
        dense_hits = self.dense.query(query, top_k=dense_top_k)
        bm25_ids = [doc_id for doc_id, _ in bm25_hits]
        dense_ids = [doc_id for doc_id, _ in dense_hits]
        fused = self.rrf.fuse(
            [bm25_ids, dense_ids],
            top_k=max(final_top_k, bm25_top_k, dense_top_k),
            return_scores=True,
        )

        if self.use_prf and fused:
            top_doc_ids = [doc_id for doc_id, _ in fused[:10]]
            rel_vectors = self.dense.embed_documents(top_doc_ids)
            query_vec = self.dense.embed_query(query)
            refined_vec = self.prf.refine(query_vec, rel_vectors)
            dense_hits_refined = self.dense.search_from_vector(refined_vec, top_k=dense_top_k)
            fused = self.rrf.fuse(
                [bm25_ids, dense_hits_refined],
                top_k=max(final_top_k, bm25_top_k, dense_top_k),
                return_scores=True,
            )

        if self.use_rerank and fused:
            doc_ids = [doc_id for doc_id, _ in fused[:final_top_k]]
            docs_text = self.bm25.get_docs(doc_ids)
            reranked = self.reranker.rerank(query, docs_text, top_n=final_top_k)
            fused_dict = dict(fused)
            final_hits = [(doc_id, fused_dict.get(doc_id, 0.0)) for doc_id, _ in reranked]
        else:
            final_hits = fused[:final_top_k]

        return final_hits
