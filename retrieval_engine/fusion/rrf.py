from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# TODO: Delete this comment if not needed anymore
"""rank_fusion.py
Usage
-----
>>> from rank_fusion import ReciprocalRankFusion
>>> bm25_hits   = ["doc3", "doc8", "doc1"]  # highest rank → index 0
>>> sbert_hits  = ["doc8", "doc2", "doc3", "doc9"]
>>> rrf = ReciprocalRankFusion(k=60)
>>> fused = rrf.fuse([bm25_hits, sbert_hits], top_n=3, return_scores=True)
>>> for doc_id, score in fused:
...     print(f"{doc_id}: {score:.4f}")

doc8: 0.0333

doc3: 0.0305

doc1: 0.0164

Compared to the original version, the *return_scores* flag makes it possible to
obtain a list of ``(doc_id, score)`` tuples instead of plain doc_ids.

API
---
class **ReciprocalRankFusion(k: int = 60)**
    | Parameter | Description                                      |
    |-----------|--------------------------------------------------|
    | k         | Controls how quickly the reciprocal term decays; |
    |           | larger *k* means flatter contribution.           |

**fuse( rankings: list[list[str]], /, *, top_n: int | None = None, return_scores: bool = False) → list[str] | list[tuple[str, float]]**
    Fuses any number of *rankings* and returns the merged result, optionally
    limited to *top_n* elements.  Set *return_scores* to *True* if you also
    want the numeric RRF scores.

MIT Licence.
"""


class ReciprocalRankFusion:
    """Reciprocal Rank Fusion (RRF) algorithm for merging ranked lists.

    Paramters:
        k (int): Constant that controls the contribution of each position.
            The RRF paper suggests to start with a value of  *k ≈ 60*.
    """

    def __init__(self, k: int = 60) -> None:
        """
        Initialize the RRF fusion with a given *k* value.

        Args:
            k (int): The constant that controls the contribution of each position.
                A larger *k* means a flatter contribution from lower-ranked documents (default: 60).
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"Parameter 'k' must be a positive integer, got {k!r}.")
        self.k = k

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def fuse(
            self,
            rankings: Sequence[Sequence[str]],
            /,
            *,
            top_k: Optional[int] = None,
            return_scores: bool = False,
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """
        Fuse multiple rankings using Reciprocal Rank Fusion (RRF).

        Args:
            rankings (Sequence[Sequence[str]]): Each element is a ranking (best at index 0).
            top_k (Optional[int]): If given, truncate the fused result to the *top_k* docs.
            return_scores (bool): If *True*, return ``(doc_id, score)`` tuples instead of just doc_ids.

        Returns:
            Union[List[str], List[Tuple[str, float]]]: The fused ranking; optionally with scores if *return_scores* is set.
        """
        # If empty rankings, return an empty list
        if not rankings:
            return []

        # Accumulate RRF scores
        scores: DefaultDict[str, float] = defaultdict(float)

        for rank in rankings:
            for position, doc_id in enumerate(rank):
                # +1 because positions are zero‑based in Python but 1‑based in RRF
                scores[doc_id] += 1.0 / (self.k + position + 1)

        # Sort by score (desc) then doc_id (asc) to ensure deterministic order
        sorted_items: List[Tuple[str, float]] = sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0])
        )

        # If top_k is specified, truncate the list
        if top_k is not None and top_k > 0 and top_k < len(sorted_items):
            sorted_items = sorted_items[:top_k]

        # When wished, return a list of (doc_id, score) tuples
        if return_scores:
            return sorted_items

        # Otherwise, return only the doc_ids
        return [doc_id for doc_id, _ in sorted_items]

    def score_dict(
            self,
            rankings: Sequence[Sequence[str]],
            /,
            *,
            normalize: bool = False
    ) -> Dict[str, float]:
        """
        Return *all* RRF scores as a mapping ``{doc_id: score}``.

        Args:
            rankings (Sequence[Sequence[str]]): Each element is a ranking (best at index 0).
            normalize (bool): If *True*, rescale scores to the range *[0,1]* so that they are easier to compare with other scoring functions.

        Returns:
            Dict[str, float]: A dictionary mapping each document ID to its RRF score. Can be normalized to the range [0, 1] if *normalize* is set to *True*.
        """
        # Call the main fusion method and retrieve doc_ids and scores as tuples
        items = self.fuse(
            rankings,
            return_scores=True
        )
        # If no items, return an empty dict
        if not items:
            return {}

        # Turn the list of tuples into a dict
        scores = {doc_id: score for doc_id, score in items}

        # When normalization is requested, rescale scores to [0, 1]
        if normalize:
            # First item has max score
            max_score = next(iter(scores.values()))
            if max_score > 0:
                scores = {doc_id: score / max_score for doc_id, score in scores.items()}
            else:
                # If max score is 0, all scores are 0, so return as is
                pass
        return scores

    def __call__(
            self,
            rankings: Sequence[Sequence[str]],
            /,
            *,
            top_n: int | None = None,
    ) -> List[str]:
        """
        Call shorthand for :py:meth:`fuse`. Returns only *doc_ids*.

        Args:
            rankings (Sequence[Sequence[str]]): Each element is a ranking (best at index 0).
            top_n (int | None): If given, truncate the fused result to the *top_n* docs.

        Returns:
            List[str]: The fused ranking, limited to *top_n* if specified.
        """
        return self.fuse(rankings, top_k=top_n, return_scores=False)
