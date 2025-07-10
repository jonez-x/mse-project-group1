import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Union
import pickle
from pathlib import Path

class DenseRetriever:
    """
    DenseRetriever uses a SentenceTransformer model to compute embeddings
    and perform cosine similarity search on normalized vectors.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the DenseRetriever with a given SentenceTransformer model.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.documents: List[str] = []
        self.embeddings: np.ndarray = None

    def fit(self, documents: List[str]) -> "DenseRetriever":
        """
        Compute normalized embeddings for the given documents.
        """
        self.documents = documents
        self.embeddings = self.model.encode(
            documents, convert_to_numpy=True, normalize_embeddings=True
        )
        return self

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float, str]]:
        """
        Return the top-k most similar documents for a given query.
        """
        scores = self.get_scores(query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(idx, float(scores[idx]), self.documents[idx]) for idx in top_indices]

    def get_scores(self, query: str) -> np.ndarray:
        """
        Compute cosine similarity scores between query and all documents.
        """
        query_vec = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        return np.dot(self.embeddings, query_vec)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save retriever state to disk using pickle.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DenseRetriever":
        """
        Load retriever state from disk.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def __getstate__(self):
        """
        Return picklable state (exclude model).
        """
        return {
            "model_name": self.model_name,
            "documents": self.documents,
            "embeddings": self.embeddings,
        }

    def __setstate__(self, state):
        """
        Restore object from pickled state.
        """
        self.model_name = state["model_name"]
        self.model = SentenceTransformer(self.model_name)
        self.documents = state["documents"]
        self.embeddings = state["embeddings"]
