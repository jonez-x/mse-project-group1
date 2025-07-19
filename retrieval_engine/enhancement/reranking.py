from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CrossEncoderReRanker:
    """
    Wrapper for a Cross-Encoder model used for re-ranking documents

    Attributes:
        tokenizer (PreTrainedTokenizer): Tokenizer instance used for encoding text.
        model (PreTrainedModel): The sequence classification model.
        batch_size (int): Batch size for processing inputs.
        normalize (bool): If True, normalize the computed scores.
        device (str): Device used for computation (e.g., "cuda", "cpu").
        entail_idx (int): Index associated with the "ENTAILMENT" label.
    """

    # Data type for model weights
    TORCH_DTYPE = torch.float16

    def __init__(
            self,
            model_name: str = "cross-encoder/nli-MiniLM2-L6-H768",
            batch_size: int = 32,
            device: Optional[str] = None,
            normalize: bool = True,
    ) -> None:
        """
        Initialize the Cross-Encoder for sequence classification tasks and set up the model and tokenizer.

        Attributes:
            tokenizer (PreTrainedTokenizer): Tokenizer instance used for encoding text.
            model (PreTrainedModel): The sequence classification model.
            batch_size (int): Batch size for processing inputs.
            normalize (bool): If True, normalize the computed scores.
            device (str): Device used for computation (e.g., "cuda", "cpu").
            entail_idx (int): Index associated with the "ENTAILMENT" label.

        Parameters:
            model_name: Default model name for the tokenizer and model. Defaults
                to "cross-encoder/nli-MiniLM2-L6-H768".
            batch_size: Batch size for processing input sequences.
            device: Device used for computation. If not specified, it is
                determined automatically based on availability.
            normalize: If True, the outputs will be normalized where appropriate.
        """

        # Initialize the Cross-Encoder model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name,
            torch_dtype=self.TORCH_DTYPE,
        )

        # Set parameters
        self.batch_size = batch_size
        self.normalize = normalize

        # Set device for model inference if not provided
        if not device:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device  # TODO: compare "mps" vs "cpu" for Mac M1 Pro
        print(f"Using device: {self.device}")

        # Move model to the specified device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        self.entail_idx = self.model.config.label2id.get("ENTAILMENT", 2)

    @torch.no_grad()
    def _score_batch(
            self,
            query: str,
            docs: Sequence[str]
    ) -> torch.Tensor:
        """
        Helper function to score a batch of documents against a query.

        This function encodes the query and documents into pairs, processes them through the model,
        and returns the scores for each document in relationto the query.
        The scores are either normalized or returned as raw logits depending on the `normalize` attribute.

        Args:
            query (str): The query string to compare against the documents.
            docs (Sequence[str]): A sequence of document strings to be scored.

        Returns:
            torch.Tensor: A tensor containing the scores for each document.
            If `normalize` is True, scores are sigmoid-normalized; otherwise,
            raw logits are returned.
        """
        # Set the pairs of query and documents for encoding
        pairs = [[query, doc] for doc in docs]

        # Encode the pairs using the tokenizer
        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        # Forward pass through the model to get logits and compute scores
        logits = self.model(**encoded).logits
        scores = logits[:, self.entail_idx]

        # Return normalized scores if specified, otherwise return raw logits
        return torch.sigmoid(scores) if self.normalize else scores

    def rerank(
            self,
            query: str,
            doc_pairs: Iterable[Tuple[str, str]],
            top_n: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Rerank a list of document pairs based on their relevance to a query and return the top N results.

        Args:
            query (str): The query string to score the documents against.
            doc_pairs (Iterable[Tuple[str, str]]): An iterable of document pairs,
                where each pair consists of a document ID and its text.
            top_n (Optional[int]): Optional; if specified, limits the results to
                the top N scored documents. If None, returns all scored documents.

        Returns:
            List[Tuple[str, float]]: A list of tuples where each tuple contains
                a document ID and its score, sorted in descending order by score.
        """
        # Ensure doc_pairs is not empty
        if not doc_pairs:
            return []

        # Unzip document pairs into IDs and texts
        doc_ids, docs = zip(*doc_pairs)

        # Initialize a list to hold scores
        scores: List[float] = []

        # Batch process the documents to score them against the query
        for i in range(0, len(docs), self.batch_size):
            batch_docs = docs[i:i + self.batch_size]
            batch_scores = self._score_batch(
                query=query,
                docs=batch_docs
            ).tolist()
            scores.extend(batch_scores)

        ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)

        # If top_n is specified, slice the ranked list to return only the top N results
        if top_n is not None and top_n < len(ranked):
            ranked = ranked[:top_n]

        # Return the ranked list, either limited to top_n or the full list
        ranked = [(doc_id, score) for doc_id, score in ranked if score > 0.0]

        return ranked
