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

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def refine(
        self,
        query_vec: np.ndarray,
        rel_doc_vecs: Sequence[np.ndarray],
        nrel_doc_vecs: Sequence[np.ndarray] | None = None,
    ) -> np.ndarray:
        """Gibt einen verfeinerten Query‑Vektor zurück.

        Parameters
        ----------
        query_vec : np.ndarray
            Ursprünglicher Query‑Embedding‑Vektor (1‑D).
        rel_doc_vecs : Sequence[np.ndarray]
            Embeddings der als relevant angenommenen Dokumente.
        nrel_doc_vecs : Sequence[np.ndarray] | None, optional
            Embeddings der als nicht‑relevant angenommenen Dokumente.

        Returns
        -------
        np.ndarray
            Neuer, normalisierter Query‑Vektor.
        """
        if not isinstance(query_vec, np.ndarray):
            query_vec = np.asarray(query_vec, dtype=np.float32)

        rel_centroid = (
            np.mean(np.asarray(rel_doc_vecs), axis=0) if rel_doc_vecs else 0
        )
        nrel_centroid = (
            np.mean(np.asarray(nrel_doc_vecs), axis=0) if nrel_doc_vecs else 0
        )

        new_vec = (
            self.alpha * query_vec + self.beta * rel_centroid - self.gamma * nrel_centroid
        )
        norm = np.linalg.norm(new_vec)
        return new_vec / norm if norm != 0 else new_vec.copy()
