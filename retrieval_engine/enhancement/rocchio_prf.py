from __future__ import annotations

from typing import Optional, Sequence, TypeAlias

import numpy as np

# Type aliases for clarity
Vector: TypeAlias = np.ndarray
VectorSequence: TypeAlias = Sequence[Vector]


class RocchioPRF:
    """
    Implementation of Rocchio Pseudo-Relevance Feedback for query expansion and refinement.

    This class implements the Rocchio algorithm to refine query vectors based on relevant
    and non-relevant document feedback.

    Attributes:
        alpha (float): Weight for the original query vector.
        beta (float): Weight for relevant document centroids.
        gamma (float): Weight for non-relevant document centroids.
    """

    # Default weights for the Rocchio algorithm
    DEFAULT_ALPHA = 1.0
    DEFAULT_BETA = 0.75
    DEFAULT_GAMMA = 0.15

    # Data type for vector conversion
    VECTOR_DTYPE = np.float32

    def __init__(
            self,
            alpha: float = DEFAULT_ALPHA,
            beta: float = DEFAULT_BETA,
            gamma: float = DEFAULT_GAMMA,
    ) -> None:
        """
        Initialize the Rocchio Pseudo-Relevance Feedback algorithm with specified weights.

        Parameters:
            alpha: Weight for the original query vector. Controls how much of the original
                query is preserved in the refined query (default: 1.0).
            beta: Weight for relevant document centroids. Controls the influence of relevant
                documents on query expansion (default: 0.75).
            gamma: Weight for non-relevant document centroids. Controls how much non-relevant
                documents are used to refine away from irrelevant terms (default: 0.15).
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _validate_inputs(
            self,
            query_vec: Vector,
            rel_doc_vecs: VectorSequence
    ) -> None:
        """
        Validate input parameters.

        Args:
            query_vec (Vector): The original query vector to be refined.
            rel_doc_vecs (VectorSequence): A sequence of document vectors considered relevant to the query.

        Raises:
            ValueError: If the query vector is empty or if no relevant document vectors are provided.
        """

        if query_vec.size == 0:
            raise ValueError("Query vector cannot be empty")
        if not rel_doc_vecs:
            raise ValueError("At least one relevant document vector is required")

    def _compute_centroid(self, vectors: VectorSequence) -> Vector:
        """
        Compute centroid vector from a sequence of vectors.

        Args:
            vectors (VectorSequence): A sequence of vectors to compute the centroid from.

        Returns:
            Vector: The centroid vector, which is the mean of the input vectors.
        """
        arr = np.asarray(vectors, dtype=self.VECTOR_DTYPE)
        return np.mean(arr, axis=0) if arr.size else np.zeros_like(arr)

    def _normalize_vector(self, vector: Vector) -> Vector:
        """
        Normalize vector using L2 norm.

        Args:
            vector (Vector): The vector to normalize.

        Returns:
            Vector: The normalized vector. If the norm is zero, returns the original vector.
        """
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def refine(
            self,
            query_vec: Vector,
            rel_doc_vecs: VectorSequence,
            nrel_doc_vecs: Optional[VectorSequence] = None
    ) -> Vector:
        """
        Refine a query vector using the Rocchio algorithm based on relevant and non-relevant document feedback.

        This method implements the classic Rocchio formula to adjust the original query vector
        by adding relevant document centroids and subtracting non-relevant document centroids.

        Args:
            query_vec (Vector): The original query vector to be refined.
            rel_doc_vecs (VectorSequence): A sequence of document vectors considered relevant to the query.
                Used to expand the query towards relevant terms.
            nrel_doc_vecs (Optional[VectorSequence]): Optional sequence of document vectors considered
                non-relevant to the query. Used to move the query away from irrelevant terms. 
                If not provided, no non-relevant feedback is applied.

        Returns:
            Vector: The refined and normalized query vector.
                The returned vector is L2-normalized unless the norm is zero,
                in which case the unnormalized vector is returned.
        """
        # Make sure input vectors are numpy arrays of the correct type
        query_vec = np.asarray(query_vec, dtype=self.VECTOR_DTYPE)
        self._validate_inputs(
            query_vec=query_vec,
            rel_doc_vecs=rel_doc_vecs,
        )

        # Compute centroids
        rel_centroid = self._compute_centroid(vectors=rel_doc_vecs)
        nrel_centroid = self._compute_centroid(vectors=nrel_doc_vecs) if nrel_doc_vecs else np.zeros_like(query_vec)

        # Apply the Rocchio formula
        refined_vector = (
                self.alpha * query_vec +
                self.beta * rel_centroid -
                self.gamma * nrel_centroid
        )

        return self._normalize_vector(vector=refined_vector)
