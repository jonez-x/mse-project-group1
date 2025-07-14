from __future__ import annotations
from typing import List, Sequence, Tuple
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from .rrf import ReciprocalRankFusion
from .rocchio_prf import RocchioPRF
from .cross_encoder_reranker import CrossEncoderReRanker


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

        dense_hits = self.dense.query(query, top_k=dense_top_k)
        bm25_ids, _ = self.bm25.query(query, top_k=bm25_top_k)
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
