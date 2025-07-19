from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np


class BM25Retriever:
    """Lightweight BM25 retriever with optional NumPy acceleration.

    Parameters
    ----------
    k1, b : float
        Standard BM25 parameters (see Robertson & Walker, 1994).
    use_numpy : bool, default ``True``
        Whether to build a dense BM25 matrix for fast matrix‑vector queries.

    Notes
    -----
    * **Corpus is kept in‑memory** (``self._corpus``) so we can return raw
      documents later.
    * All tokens are lower‑cased alphanumerics extracted via ``re.findall``.
    * The API mirrors our earlier stub, but fixes a missing attribute and
      tightens type hints.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75, use_numpy: bool = True):
        self.k1: float = k1
        self.b: float = b
        self.use_numpy: bool = use_numpy

        # corpus‑level statistics – filled in ``fit``
        self._corpus: List[str] = []
        self.terms: List[List[str]] = []
        self.doc_len: List[int] = []
        self.avg_len: float = 0.0
        self.doc_freq: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._num_docs: int = 0

        # matrix representation (optional)
        self._matrix: np.ndarray | None = None
        self._vocab_idx: Dict[str, int] | None = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Very simple tokenizer – lower‑case alphanumerics."""
        return re.findall(r"\w+", text.lower())

    def fit(self, docs: Sequence[str]) -> "BM25Retriever":
        """Index *docs* and pre‑compute IDF / matrix statistics."""
        # Keep corpus for later retrieval
        self._corpus = list(docs)  # ensures random‑access by integer ID

        # Tokenise once
        self.terms = [self._tokenize(d) for d in self._corpus]

        # Basic stats
        self.doc_len = [len(toks) for toks in self.terms]
        self._num_docs = len(self.doc_len)
        self.avg_len = float(sum(self.doc_len)) / max(self._num_docs, 1)

        # Document frequency counts
        for toks in self.terms:
            for t in set(toks):
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1

        # Inverse document frequency
        N = self._num_docs
        self.idf = {t: np.log(1 + (N - df + 0.5) / (df + 0.5)) for t, df in self.doc_freq.items()}

        if self.use_numpy:
            self._build_matrix()
        return self

    def _build_matrix(self) -> None:
        vocab = list(self.idf)
        self._vocab_idx = {t: i for i, t in enumerate(vocab)}
        mat = np.zeros((self._num_docs, len(vocab)), dtype=np.float32)

        for row, toks in enumerate(self.terms):
            counts: Dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            L = self.doc_len[row]
            for t, tf in counts.items():
                col = self._vocab_idx[t]
                idf = self.idf[t]
                mat[row, col] = idf * (tf * (self.k1 + 1)) / (
                        tf + self.k1 * (1 - self.b + self.b * L / self.avg_len)
                )
        self._matrix = mat

    def _score_doc(self, query_tokens: List[str], doc_idx: int) -> float:
        L = self.doc_len[doc_idx]
        score = 0.0
        tf_counts: Dict[str, int] = {}
        for tok in self.terms[doc_idx]:
            tf_counts[tok] = tf_counts.get(tok, 0) + 1
        for t in query_tokens:
            idf = self.idf.get(t)
            if idf is None:
                continue
            tf = tf_counts.get(t, 0)
            score += idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * L / self.avg_len)
            )
        return score

    def query(self, text: str, *, top_k: int = 10, return_scores: bool = False) -> Tuple[
        List[int], List[float]]:  # noqa: D401,E501
        """Return *indices* of the ``top_k`` most relevant docs for *text*."""
        q_tokens = self._tokenize(text)
        if self.use_numpy and self._matrix is not None and self._vocab_idx is not None:
            q_vec = np.zeros((len(self._vocab_idx),), dtype=np.float32)
            for tok in q_tokens:
                col = self._vocab_idx.get(tok)
                if col is not None:
                    q_vec[col] = self.idf[tok]
            scores = self._matrix @ q_vec
        else:
            scores = np.array([self._score_doc(q_tokens, i) for i in range(self._num_docs)], dtype=np.float32)

        top_idx = np.argsort(-scores)[:top_k]
        if return_scores:
            return top_idx.tolist(), scores[top_idx].tolist()
        return top_idx.tolist(), []

    def get_docs(self, doc_ids: Sequence[int]) -> List[Tuple[str, str]]:
        """Return *[(doc_id, raw_text)]* for every ID in *doc_ids*.

        Raises ``IndexError`` if an ID is out of range.
        """
        pairs: List[Tuple[str, str]] = []
        for doc_id in doc_ids:
            if not 0 <= doc_id < self._num_docs:
                raise IndexError(f"DocID {doc_id} is out of range 0 … {self._num_docs - 1}")
            pairs.append((str(doc_id), self._corpus[doc_id]))
        return pairs

    def save(self, path: Union[str, Path]) -> None:
        """Pickle the entire retriever to *path*."""
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BM25Retriever":
        """Load a previously pickled retriever from *path*."""
        with open(path, "rb") as fh:
            return pickle.load(fh)
