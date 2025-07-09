import math
import pickle
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np


class BM25Indexer:
    def __init__(self, k1: float = 1.5, b: float = 0.75, use_numpy: bool = True):
        self.k1 = k1
        self.b = b
        self.use_numpy = use_numpy
        self.doc_len: List[int] = []
        self.avg_len = 0.0
        self.doc_freq: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.terms: List[List[str]] = []
        self._num_docs = 0

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def fit(self, docs: Sequence[str]) -> "BM25Indexer":
        self.terms = [self._tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.terms]
        self.avg_len = sum(self.doc_len) / max(len(self.doc_len), 1)
        self._num_docs = len(docs)
        for tokens in self.terms:
            for token in set(tokens):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
        self.idf = {
            t: math.log(1 + (self._num_docs - df + 0.5) / (df + 0.5))
            for t, df in self.doc_freq.items()
        }
        if self.use_numpy:
            self._build_matrix()
        return self

    def _build_matrix(self):
        vocab = list(self.idf)
        index = {t: i for i, t in enumerate(vocab)}
        mat = np.zeros((len(self.terms), len(vocab)), dtype=np.float32)
        for row, tokens in enumerate(self.terms):
            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            L = self.doc_len[row]
            for t, freq in counts.items():
                col = index[t]
                idf = self.idf[t]
                score = idf * (freq * (self.k1 + 1)) / (
                    freq + self.k1 * (1 - self.b + self.b * L / self.avg_len)
                )
                mat[row, col] = score
        self._matrix = mat
        self._vocab_idx = index

    def _score_doc(self, tokens: List[str], doc_index: int) -> float:
        freq: Dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        L = self.doc_len[doc_index]
        score = 0.0
        for t, qf in freq.items():
            idf = self.idf.get(t)
            if idf is None:
                continue
            tf = self.terms[doc_index].count(t)
            score += idf * (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * L / self.avg_len)
            )
        return score

    def query(
        self, text: str, top_n: int = 10, return_scores: bool = False
    ) -> Tuple[List[int], List[float]]:
        tokens = self._tokenize(text)
        if self.use_numpy and hasattr(self, "_matrix"):
            q_vec = np.zeros((len(self._vocab_idx),), dtype=np.float32)
            for t in tokens:
                idx = self._vocab_idx.get(t)
                if idx is not None:
                    q_vec[idx] = self.idf[t]
            scores = self._matrix @ q_vec
        else:
            scores = np.array(
                [self._score_doc(tokens, i) for i in range(len(self.terms))],
                dtype=np.float32,
            )
        top_idx = np.argsort(-scores)[:top_n]
        if return_scores:
            return top_idx.tolist(), scores[top_idx].tolist()
        return top_idx.tolist(), []

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BM25Indexer":
        with open(path, "rb") as f:
            return pickle.load(f)
