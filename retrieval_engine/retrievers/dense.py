from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval_engine.docs.document_store import Document


class DenseRetriever:
    """
    SBERT-based semantic retriever using dense vector representations and cosine similarity.

    This class implements dense retrieval using pre-trained sentence transformer models
    to encode documents and queries into high-dimensional vector spaces. It performs
    similarity search using cosine similarity between query and document embeddings.

    Attributes:
        model (SentenceTransformer): The sentence transformer model for encoding text.
        normalize (bool): Whether to L2-normalize embeddings for cosine similarity.
        batch_size (int | None): Batch size for encoding operations.
        _corpus (List[str]): The original document texts kept in memory.
        _doc_store_map (List[Document]): The original Document instances.
        _embeddings (np.ndarray | None): Pre-computed document embeddings matrix.
    """

    def __init__(
            self,
            model_name: str = "all-MiniLM-L6-v2",
            normalize_embeddings: bool = True,
            batch_size: Optional[int] = None,
            device: Optional[str] = None,
    ) -> None:
        """
        Initialize the dense retriever with a sentence transformer model.

        Parameters:
            model_name: Name or path of the sentence transformer model to use (default: "all-MiniLM-L6-v2").
            normalize_embeddings: Whether to L2-normalize embeddings for cosine similarity.
                Recommended for most use cases (default: True).
            batch_size: Batch size for encoding operations. If None, uses model default.
                Larger batches are faster but use more memory.
            device: Device to run the model on ("cuda", "cpu", etc.). If None,
                automatically selects the best available device.
        """
        # Initialize the sentence transformer model
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize_embeddings
        self.batch_size = batch_size

        # Internal state for fitted corpus
        self._corpus: List[str] = []
        self._doc_store_map: List[Document] = []
        self._embeddings: Optional[np.ndarray] = None

    def fit(self, corpus: Sequence[Document]) -> None:
        """
        Encode the document corpus and store embeddings in memory for fast retrieval.

        Args:
            corpus: A sequence of Document objects to encode and index.
        """
        self._doc_store_map = list(corpus)
        self._corpus = [doc.to_text() for doc in self._doc_store_map]

        # Encode all documents using the sentence transformer model
        self._embeddings = self.model.encode(
            self._corpus,
            batch_size=self.batch_size or 32,  # If no default batch size is set, use 32
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        # Normalize embeddings for cosine similarity if enabled
        if self.normalize:
            self._embeddings = self._l2_normalize(self._embeddings)

    def query(
            self,
            query: str,
            top_k: int = 100
    ) -> List[Tuple[int, float, Document]]:
        """
        Search for the most similar documents to a query string.

        Args:
            query: The search query string.
            top_k: Number of most similar documents to return (default: 100).

        Returns:
            List[Tuple[int, float, Document]]: A list of tuples containing:
                - Document index (int)
                - Cosine similarity score (float)
                - Original Document object
                Sorted by similarity score in descending order.
        """
        query_vec = self.embed_query(query)
        return self.search_from_vector(query_vec, top_k)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encode a query string into a dense vector representation.

        Args:
            query: The query string to encode.

        Returns:
            np.ndarray: The dense vector representation of the query.
                L2-normalized if normalize_embeddings is True.
        """
        vec = self.model.encode(query, convert_to_numpy=True)
        return self._l2_normalize(vec) if self.normalize else vec

    def embed_documents(self, docs: Sequence[str]) -> np.ndarray:
        """
        Encode arbitrary document texts into dense vector representations.

        Args:
            docs: A sequence of document strings to encode.

        Returns:
            np.ndarray: A matrix of dense vectors with shape (num_docs, embedding_dim).
                L2-normalized if normalize_embeddings is True.
        """
        vecs = self.model.encode(
            list(docs),
            batch_size=self.batch_size or 32,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._l2_normalize(vecs) if self.normalize else vecs

    def search_from_vector(
            self,
            query_vec: np.ndarray,
            top_k: int = 100
    ) -> List[Tuple[str, float, Document]]:
        """
        Perform similarity search using a pre-computed query vector.

        Args:
            query_vec: Pre-computed query vector to search with.
            top_k: Number of most similar documents to return (default: 100).

        Returns:
            List[Tuple[str, float, Document]]: A list of tuples containing:
                - Document ID (str, typically URL)
                - Cosine similarity score (float)
                - Original Document object
            Sorted by similarity score in descending order.

        Raises:
            ValueError: If the retriever has not been fitted with a corpus.
        """
        if self._embeddings is None:
            raise ValueError("DenseRetriever not fitted with a corpus. Call fit() first.")

        sims = np.dot(self._embeddings, query_vec)

        idx = np.argpartition(-sims, top_k)[:top_k]
        sorted_idx = idx[np.argsort(-sims[idx])]

        return [(self._doc_store_map[i].url, float(sims[i]), self._doc_store_map[i]) for i in sorted_idx]

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[int, float, Document]]:
        """
        Alias for query method to maintain backward compatibility with tests.

        Args:
            query: The search query string.
            top_k: Number of most similar documents to return (default: 100).

        Returns:
            List[Tuple[int, float, Document]]: A list of tuples containing:
                - Document index (int)
                - Cosine similarity score (float)
                - Original Document object
                Sorted by similarity score in descending order.
        """
        return self.query(query, top_k)

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        """
        Helper function to L2-normalize vectors for cosine similarity computation.

        Args:
            x: Input array to normalize. Can be 1D (single vector) or 2D (batch of vectors).

        Returns:
            np.ndarray: L2-normalized array with the same shape as input.
                Zero vectors are left unchanged to avoid division by zero.
        """
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        # Avoid division by zero for zero vectors
        norm[norm == 0] = 1.0
        return x / norm
