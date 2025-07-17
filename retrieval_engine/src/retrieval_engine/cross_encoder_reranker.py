from __future__ import annotations
from typing import Iterable, List, Sequence, Tuple

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
except ImportError as err:  # pragma: no cover
    raise ImportError(
        "CrossEncoderReRanker requires the 'transformers' and 'torch' packages. "
        "Install them via `pip install transformers torch`."
    ) from err


class CrossEncoderReRanker:
    """Lightweight wrapper around a Cross‑Encoder for re‑ranking."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-MiniLM2-L6-H768",
        batch_size: int = 32,
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float16)
        self.batch_size = batch_size
        self.normalize = normalize

        if device is None:
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device        # TODO: compare "mps" vs "cpu" for Mac M1 Pro
        print(f"Using device: {self.device}")

        self.model.to(self.device)
        self.model.eval()
        self.entail_idx = self.model.config.label2id.get("ENTAILMENT", 2)

    @torch.no_grad()
    def _score_batch(self, query: str, docs: Sequence[str]) -> torch.Tensor:
        pairs = [[query, doc] for doc in docs]
        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**encoded).logits
        scores = logits[:, self.entail_idx]
        return torch.sigmoid(scores) if self.normalize else scores

    def rerank(
            self,
            query: str,
            doc_pairs: Iterable[Tuple[str, str]],
            top_n: int | None = None,
    ) -> List[Tuple[str, float]]:
        """Return *(doc_id, score)* sorted descending by score."""
        doc_ids, docs = zip(*doc_pairs) if doc_pairs else ([], [])
        scores: List[float] = []

        for i in range(0, len(docs), self.batch_size):
            batch_docs = docs[i:i + self.batch_size]
            batch_scores = self._score_batch(query, batch_docs)
            scores.extend(batch_scores.tolist())

        ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked if top_n is None else ranked[:top_n]
