import numpy as np
import pickle
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union


class BM25Indexer:
    """
    BM25Indexer class provides functionality for indexing and querying documents based on the BM25 algorithm.

    Attributes:
        k1 (float): Term frequency saturation parameter.
        b (float): Document length normalization parameter.
        use_numpy (bool): Determines whether NumPy is used for matrix operations.
        doc_len (List[int]): Stores the length of each document in the corpus.
        avg_len (float): Represents the average document length in the corpus.
        doc_freq (Dict[str, int]): Maps terms to their document frequencies.
        idf (Dict[str, float]): Stores inverse document frequencies for each term.
        terms (List[List[str]]): A list of tokenized documents.
    """

    def __init__(
            self,
            k1: float = 1.5,
            b: float = 0.75,
            use_numpy: bool = True,
    ):
        """
        Initialize the BM25 indexer.

        Args:
            k1 (float): Term frequency saturation parameter.
            b (float): Document length normalization parameter.
            use_numpy (bool): Whether to use NumPy for matrix operations. Else, use pure Python.
        """
        # Validate parameters
        self.k1: float = k1
        self.b: float = b
        self.use_numpy: bool = use_numpy

        # Initialize attributes
        self.doc_len: List[int] = []
        self.avg_len: float = 0.0
        self.doc_freq: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.terms: List[List[str]] = []
        self._num_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """
        Helper method to tokenize a string into words.

        Args:
            text (str): The input string to tokenize.
        Returns:
            List[str]: A list of lowercase words extracted from the input string.
        """

        return re.findall(r"\w+", text.lower())

    def fit(self, docs: Sequence[str]) -> "BM25Indexer":
        """
        Fit the indexer to a sequence of documents and compute neccessary statistics (like IDF and document lengths).

        Args:
            docs (Sequence[str]): A sequence of documents to index.
        Returns:
            BM25Indexer: The fitted indexer instance.
        """
        # Tokenize documents
        self.terms = [self._tokenize(text=d) for d in docs]

        # Calculate document lengths and average length
        self.doc_len = [len(t) for t in self.terms]
        self.avg_len = sum(self.doc_len) / max(len(self.doc_len), 1)
        self._num_docs = len(docs)

        # Go through terms to calculate document frequency and IDF
        for tokens in self.terms:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                # Increment document frequency for each unique token
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

        # Calculate IDF for each term using the BM25 formula:
        #  IDF(t) = log(1 + (N - df + 0.5) / (df + 0.5))
        self.idf = {
            t: np.log(1 + (self._num_docs - df + 0.5) / (df + 0.5))
            for t, df in self.doc_freq.items()
        }

        # When using NumPy, build the matrix representation of the BM25 scores
        if self.use_numpy:
            self._build_matrix()
        return self

    def _build_matrix(self):
        """
        Helper method to build the BM25 score matrix using NumPy. Only called if use_numpy is True.
        """
        # Create vocabulary list from all terms with computed IDF values
        vocab = list(self.idf)
        # Create mapping from term to column index in the matrix
        index = {t: i for i, t in enumerate(vocab)}

        # Initialize matrix: rows = documents, columns = vocabulary terms
        mat = np.zeros((len(self.terms), len(vocab)), dtype=np.float32)

        # Process each document to fill the matrix
        for row, tokens in enumerate(self.terms):
            # Count term frequencies in the current document
            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1

            # Get document length for normalization
            L = self.doc_len[row]

            # Calculate BM25 score for each term in this document
            for t, freq in counts.items():
                col = index[t]  # Get column index for this term
                idf = self.idf[t]  # Get pre-computed IDF value

                # Apply BM25 formula:
                #  IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * L / avg_len))
                score = idf * (freq * (self.k1 + 1)) / (
                        freq + self.k1 * (1 - self.b + self.b * L / self.avg_len)
                )
                mat[row, col] = score

        # Store the computed matrix and vocabulary index for later use
        self._matrix = mat
        self._vocab_idx = index

    def _score_doc(
            self,
            tokens: List[str],
            doc_index: int,
    ) -> float:
        """
        Calculate the BM25 score for a specific document given query tokens.

        Args:
            tokens (List[str]): Tokenized query terms.
            doc_index (int): Index of the document to score.

        Returns:
            float: The BM25 score for the document.
        """
        # Count frequency of each query token
        freq: Dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1

        # Get document length
        L = self.doc_len[doc_index]
        score = 0.0

        # Calculate BM25 score for each query term
        for t, qf in freq.items():
            idf = self.idf.get(t)
            if idf is None:
                continue  # Skip terms not in corpus

            # Count term frequency in the document
            tf = self.terms[doc_index].count(t)

            # Apply BM25 formula: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * L / avg_len))
            score += idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * L / self.avg_len)
            )
        return score

    def query(
            self,
            text: str,
            top_n: int = 10,
            return_scores: bool = False,
    ) -> Tuple[List[int], List[float]]:
        """
        Query the indexed documents and return the top-N most relevant documents.

        Args:
            text (str): The query string.
            top_n (int): Number of top documents to return. Defaults to 10.
            return_scores (bool): Whether to return the scores along with document indices. Defaults to False.

        Returns:
            Tuple[List[int], List[float]]: A tuple containing:
                - List of document indices ranked by relevance
                - List of corresponding scores (empty if return_scores is False)
        """
        # Tokenize the query
        tokens = self._tokenize(text=text)

        # Use matrix-based computation if available (as it's faster)
        if self.use_numpy and hasattr(self, "_matrix"):
            # Create query vector with IDF weights
            q_vec = np.zeros((len(self._vocab_idx),), dtype=np.float32)
            for t in tokens:
                idx = self._vocab_idx.get(t)
                if idx is not None:
                    q_vec[idx] = self.idf[t]
            # Matrix multiplication to get scores for all documents
            scores = self._matrix @ q_vec

        else:
            # Fall back to document-by-document scoring
            scores = np.array(
                [self._score_doc(tokens, i) for i in range(len(self.terms))],
                dtype=np.float32,
            )

        # Get top N documents by score (descending order)
        top_idx = np.argsort(-scores)[:top_n]

        if return_scores:
            return top_idx.tolist(), scores[top_idx].tolist()

        return top_idx.tolist(), []

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the BM25 indexer to a file using pickle.

        Args:
            path (str): The file path where the indexer will be saved.
        """
        # Serialize the entire indexer object to disk
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BM25Indexer":
        """
        Load a BM25 indexer from a file.

        Args:
            path (str): The file path from which to load the indexer.

        Returns:
            BM25Indexer: The loaded indexer instance.
        """
        # Deserialize the indexer object from disk
        with open(path, "rb") as f:
            return pickle.load(f)
