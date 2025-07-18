from __future__ import annotations
from typing import Sequence
import numpy as np

class RocchioPRF:
    """Pseudo‑Relevance Feedback nach Rocchio.

    Parameters
    ----------
    alpha : float, optional
        Gewicht des ursprünglichen Query‑Vektors (Default 1.0).
    beta : float, optional
        Gewicht der (angenommen) relevanten Dokumente (Default 0.75).
    gamma : float, optional
        Gewicht der (angenommen) nicht‑relevanten Dokumente (Default 0.15).
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.75, gamma: float = 0.15) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def refine(
            self,
            query_vec: np.ndarray,
            rel_doc_vecs: Sequence[np.ndarray],
            nrel_doc_vecs: Sequence[np.ndarray] | None = None,
    ) -> np.ndarray:
        if not isinstance(query_vec, np.ndarray):
            query_vec = np.asarray(query_vec, dtype=np.float32)

        rel_arr = np.asarray(rel_doc_vecs, dtype=np.float32)
        rel_centroid = np.mean(rel_arr, axis=0) if rel_arr.size else 0

        if nrel_doc_vecs is not None:
            nrel_arr = np.asarray(nrel_doc_vecs, dtype=np.float32)
            nrel_centroid = np.mean(nrel_arr, axis=0) if nrel_arr.size else 0
        else:
            nrel_centroid = 0

        new_vec = (
                self.alpha * query_vec
                + self.beta * rel_centroid
                - self.gamma * nrel_centroid
        )
        norm = np.linalg.norm(new_vec)
        return new_vec / norm if norm else new_vec

