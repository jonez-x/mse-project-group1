from __future__ import annotations
from typing import List, Sequence, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    """SBERT‑basierter semantischer Retriever (Cosine Similarity)."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        normalize_embeddings: bool = True,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> None:
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize_embeddings
        self.batch_size = batch_size

        self._corpus: List[str] = []
        self._embeddings: np.ndarray | None = None  # shape (N, dim)

    def fit(self, corpus: Sequence[str]) -> None:
        """Encode the full *corpus* and keep embeddings in memory."""
        self._corpus = list(corpus)
        self._embeddings = self.model.encode(
            self._corpus,
            batch_size=self.batch_size or 32,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if self.normalize:
            self._embeddings = self._l2_normalize(self._embeddings)

    def query(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """Return top‑k doc indices with cosine‑similarity scores."""
        query_vec = self.embed_query(query)
        return self.search_from_vector(query_vec, top_k)

    def embed_query(self, query: str) -> np.ndarray:
        """Return *normalized* SBERT embedding for the query."""
        vec = self.model.encode(query, convert_to_numpy=True)
        return self._l2_normalize(vec) if self.normalize else vec

    def embed_documents(self, docs: Sequence[str]) -> np.ndarray:
        """Encode *arbitrary* document texts (no internal IDs needed)."""
        vecs = self.model.encode(
            list(docs),
            batch_size=self.batch_size or 32,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._l2_normalize(vecs) if self.normalize else vecs

    def search_from_vector(
        self, query_vec: np.ndarray, top_k: int = 100
    ) -> List[Tuple[int, float]]:
        """Compute similarities from a *given* query vector (e.g. Rocchio PRF)."""
        if self._embeddings is None:
            raise ValueError("DenseRetriever not fitted – call .fit() first.")

        sims = np.dot(self._embeddings, query_vec)
        idx = np.argpartition(-sims, top_k)[:top_k]
        sorted_idx = idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[i])) for i in sorted_idx]

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        norm[norm == 0] = 1.0
        return x / norm

