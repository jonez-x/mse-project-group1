import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Union, Optional


class DenseRetriever:
    """
    DenseRetriever class provides functionality for dense vector-based document retrieval using SentenceTransformer models.

    This class uses pre-trained transformer models to encode documents and queries into dense vector representations,
    then performs similarity search using cosine similarity on normalized embeddings.

    Attributes:
        model_name (str): Name of the SentenceTransformer model used for encoding.
        model (SentenceTransformer): The loaded SentenceTransformer model instance.
        documents (List[str]): List of documents that have been indexed.
        embeddings (np.ndarray): Normalized embeddings for all indexed documents.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the DenseRetriever with a specified SentenceTransformer model.

        Args:
            model_name (str): Name of the SentenceTransformer model to use for encoding.
                            Defaults to "all-MiniLM-L6-v2", a lightweight and efficient model.
        """
        self.model_name = model_name

        # Initialize storage for documents and their embeddings
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

        # Load the SentenceTransformer model
        self.model = SentenceTransformer(model_name_or_path=model_name)

    def fit(self, documents: List[str]) -> "DenseRetriever":
        """
        Fit the retriever to a collection of documents by computing their embeddings.

        This method encodes all provided documents into normalized dense vector representations
        using the configured SentenceTransformer model. The embeddings are stored for later
        similarity search operations.

        Args:
            documents (List[str]): A list of documents to index for retrieval.

        Returns:
            DenseRetriever: The fitted retriever instance (for method chaining).
        """
        # Store the documents for later reference
        self.documents = documents
        # Compute normalized embeddings for all documents
        self.embeddings = self.model.encode(
            sentences=documents,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return self

    def retrieve(
            self,
            query: str,
            top_k: int = 10,
    ) -> List[Tuple[int, float, str]]:
        """
        Retrieve the most similar documents for a given query using cosine similarity.

        This method encodes the query, computes similarity scores against all indexed documents,
        and returns the top-k most relevant documents ranked by similarity score.

        Args:
            query (str): The search query string.
            top_k (int): Number of top documents to return. Defaults to 10.

        Returns:
            List[Tuple[int, float, str]]: A list sorted by similarity score in descending order, where each tuple contains:
                - Document index (int)
                - Similarity score (float)
                - Document text (str)
        """
        # Compute similarity scores for the query against all documents
        scores = self.get_scores(query=query)

        # Get top-k indices sorted by score in descending order
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Prepare results as tuples of (index, score, document)
        results = [(idx, float(scores[idx]), self.documents[idx]) for idx in top_indices]
        return results

    def get_scores(self, query: str) -> np.ndarray:
        """
        Compute cosine similarity scores between a query and all indexed documents.

        This method encodes the query into a normalized vector and computes dot product
        similarity with all document embeddings (equivalent to cosine similarity for normalized vectors).

        Args:
            query (str): The query string to compute similarities for.

        Returns:
            np.ndarray: Array of similarity scores for each document, in the same order as self.documents.
        """
        # Encode query into normalized vector
        query_vec = self.model.encode(
            sentences=[query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        # Compute dot product (cosine similarity for normalized vectors)
        return np.dot(self.embeddings, query_vec)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the DenseRetriever instance to a file using pickle serialization.

        The model state is excluded from serialization to reduce file size.
        The model will be reloaded from the model_name when the object is restored.

        Args:
            path (Union[str, Path]): The file path where the retriever should be saved.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DenseRetriever":
        """
        Load a DenseRetriever instance from a file.

        This method deserializes a previously saved DenseRetriever object and
        reconstructs the SentenceTransformer model from the stored model name.

        Args:
            path (Union[str, Path]): The file path from which to load the retriever.

        Returns:
            DenseRetriever: The loaded retriever instance.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def __getstate__(self):
        """
        Prepare object state for pickling by excluding the SentenceTransformer model.

        The model is excluded to reduce serialization size and avoid potential
        compatibility issues. It will be reconstructed during unpickling.

        Returns:
            dict: Dictionary containing the picklable state of the object.
        """
        # Return only the essential state, excluding the model object
        return {
            "model_name": self.model_name,
            "documents": self.documents,
            "embeddings": self.embeddings,
        }

    def __setstate__(self, state):
        """
        Restore object from pickled state by reconstructing the SentenceTransformer model.

        This method is called during unpickling to restore the object state and
        reload the SentenceTransformer model from the stored model name.

        Args:
            state (dict): Dictionary containing the pickled object state.
        """
        # Restore the basic attributes
        self.model_name = state["model_name"]
        # Reconstruct the SentenceTransformer model
        self.model = SentenceTransformer(self.model_name)
        # Restore the indexed documents and their embeddings
        self.documents = state["documents"]
        self.embeddings = state["embeddings"]
