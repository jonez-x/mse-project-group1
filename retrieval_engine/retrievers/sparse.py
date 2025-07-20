from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm

from retrieval_engine.docs.document_store import Document


class BM25Retriever:
    """
    Lightweight BM25 retriever with optional NumPy acceleration for efficient sparse retrieval.

    This class implements the BM25 ranking function, a probabilistic ranking function used in
    information retrieval. It supports both traditional term-by-term scoring and an optional
    NumPy-accelerated matrix-vector approach for faster queries on large corpora.

    Attributes:
        k1 (float): Controls term frequency saturation point in BM25 formula.
        b (float): Controls how much document length normalizes term frequency.
        use_numpy (bool): Whether to build a dense BM25 matrix for fast matrix-vector queries.
        _corpus (List[str]): The original document texts kept in memory for retrieval.
        _doc_store_map (List[Document]): Original Document instances for result mapping.
        terms (List[List[str]]): Tokenized version of each document in the corpus.
        doc_len (List[int]): Length of each document in terms of token count.
        avg_len (float): Average document length across the corpus.
        doc_freq (Dict[str, int]): Document frequency for each term in the vocabulary.
        idf (Dict[str, float]): Inverse document frequency for each term.
        _num_docs (int): Total number of documents in the corpus.
        _matrix (np.ndarray | None): Optional dense BM25 matrix for accelerated queries.
        _vocab_idx (Dict[str, int] | None): Mapping from terms to matrix column indices.

    Notes:
        - Corpus is kept in-memory for document retrieval functionality
        - All tokens are lower-cased alphanumerics extracted via regex
        - The NumPy acceleration builds a dense matrix which may use significant memory
        - Standard BM25 parameters follow Robertson & Walker (1994)
    """

    def __init__(
            self,
            *,
            k1: float = 1.5,
            b: float = 0.75,
            use_numpy: bool = True
    ) -> None:
        """
        Initialize the BM25 retriever with specified parameters.

        Parameters:
            k1: Controls term frequency saturation point. Higher values give more weight
                to term frequency (default: 1.5).
            b: Controls document length normalization. 0 = no normalization, 1 = full
                normalization (default: 0.75).
            use_numpy: Whether to build a dense BM25 matrix for fast matrix-vector queries.
                Trades memory for query speed (default: True).
        """
        self.k1: float = k1
        self.b: float = b
        self.use_numpy: bool = use_numpy

        self._corpus: List[str] = []
        self._doc_store_map: List[Document] = []
        self.terms: List[List[str]] = []
        self.doc_len: List[int] = []
        self.avg_len: float = 0.0
        self.doc_freq: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._num_docs: int = 0

        self._matrix: Optional[np.ndarray] = None
        self._vocab_idx: Optional[Dict[str, int]] = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Simple tokenizer that extracts lower-case alphanumeric tokens.

        Args:
            text: The input text to tokenize.

        Returns:
            List[str]: A list of lowercase alphanumeric tokens extracted from the text.
        """
        return re.findall(r"\w+", text.lower())

    def fit(self, docs: Sequence[Document]) -> "BM25Retriever":
        """
        Index the document collection and pre-compute BM25 statistics.

        This method processes the input documents by tokenizing them, computing document
        frequencies, calculating IDF values, and optionally building a dense matrix
        representation for accelerated queries.

        Args:
            docs: A sequence of Document objects to index.

        Returns:
            BM25Retriever: Returns self for method chaining.
        """
        # Keep documents for retrieval
        self._doc_store_map = list(docs)

        # Store corpus as pure text representation
        self._corpus = [doc.to_text() for doc in self._doc_store_map]

        self.terms = [self._tokenize(text=d) for d in tqdm(self._corpus, desc="Tokenizing")]

        # Compute basic document statistics
        self.doc_len = [len(toks) for toks in self.terms]
        self._num_docs = len(self.doc_len)
        self.avg_len = float(sum(self.doc_len)) / max(self._num_docs, 1)

        # Count document frequency for each term
        for toks in self.terms:
            for t in set(toks):  # Use set to count each term only once per document
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1

        # Compute inverse document frequency using BM25 IDF formula
        N = self._num_docs
        self.idf = {
            t: np.log(1 + (N - df + 0.5) / (df + 0.5))
            for t, df in self.doc_freq.items()
        }

        # Build dense matrix representation if NumPy acceleration is enabled
        if self.use_numpy:
            self._build_matrix()
        return self

    def _build_matrix(self) -> None:
        """
        Build a dense BM25 matrix for accelerated query processing.
        """
        # Create vocabulary and index mapping
        vocab = list(self.idf)
        self._vocab_idx = {t: i for i, t in enumerate(vocab)}

        # Initialize the BM25 matrix (documents x terms)
        mat = np.zeros((self._num_docs, len(vocab)), dtype=np.float32)

        for row, toks in tqdm(enumerate(self.terms), total=self._num_docs, desc="Building BM25 matrix"):
            counts: Dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1

            # Get document length for normalization
            L = self.doc_len[row]

            # Compute BM25 score for each term in this document
            for t, tf in counts.items():
                col = self._vocab_idx[t]
                idf = self.idf[t]
                # Apply BM25 formula: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * L / avg_len))
                mat[row, col] = idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * L / self.avg_len)
                )

        self._matrix = mat

    def _score_doc(
            self,
            query_tokens: List[str],
            doc_idx: int
    ) -> float:
        """
        Compute BM25 score for a single document against query tokens. Used when NumPy acceleration is not enabled.

        Args:
            query_tokens: List of tokenized query terms.
            doc_idx: Index of the document to score.

        Returns:
            float: The BM25 score for the document.
        """
        L = self.doc_len[doc_idx]
        score = 0.0

        # Count term frequencies in the document
        tf_counts: Dict[str, int] = {}
        for tok in self.terms[doc_idx]:
            tf_counts[tok] = tf_counts.get(tok, 0) + 1

        # Sum BM25 contributions for each query term
        for t in query_tokens:
            idf = self.idf.get(t)
            if idf is None:  # Skip terms not in vocabulary
                continue
            tf = tf_counts.get(t, 0)
            # Apply BM25 formula
            score += idf * (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * L / self.avg_len)
            )
        return score

    def query(
            self,
            query: str,
            *,
            top_k: int = 10,
            return_scores: bool = False
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """
        Search for the most relevant documents given a query text. Tokenizes the query,
        computes BM25 scores, and returns the top-k documents sorted by relevance.

        Args:
            query: The query string to search for.
            top_k: Number of top documents to return (default: 10).
            return_scores: Whether to return the BM25 scores along with indices (default: False).

        Returns:
            List[Document] or List[Tuple[Document, float]] depending on return_scores.
        """
        q_tokens = self._tokenize(query)

        if self.use_numpy and self._matrix is not None and self._vocab_idx is not None:
            q_vec = np.zeros((len(self._vocab_idx),), dtype=np.float32)
            for tok in q_tokens:
                col = self._vocab_idx.get(tok)
                if col is not None:
                    q_vec[col] = self.idf[tok]
            scores = self._matrix @ q_vec
        else:
            scores = np.array([
                self._score_doc(q_tokens, i) for i in range(self._num_docs)
            ], dtype=np.float32)

        top_idx = np.argsort(-scores)[:top_k]

        if return_scores:
            return [(self._doc_store_map[i], float(scores[i])) for i in top_idx]

        return [self._doc_store_map[i] for i in top_idx]

    def get_docs(self, doc_ids: Sequence[int]) -> List[Document]:
        """
        Retrieve the original Document objects for given document IDs.

        Args:
            doc_ids: A sequence of document indices to retrieve.

        Returns:
            List[Document]: A list of Document instances.

        Raises:
            IndexError: If any document ID is out of range.
        """
        docs: List[Document] = []
        for doc_id in doc_ids:
            # Validate document ID is within valid range
            if not 0 <= doc_id < self._num_docs:
                raise IndexError(f"DocID {doc_id} is out of range 0 … {self._num_docs - 1}")
            docs.append(self._doc_store_map[doc_id])
        return docs

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the entire retriever state to disk using pickle.

        Args:
            path: File path where the retriever should be saved.
        """
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BM25Retriever":
        """
        Load a previously saved retriever from disk.

        Args:
            path: File path from which to load the retriever.

        Returns:
            BM25Retriever: The loaded retriever instance with all fitted data.
        """
        with open(path, "rb") as fh:
            return pickle.load(fh)
