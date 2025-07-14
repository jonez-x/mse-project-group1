from __future__ import annotations

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

from collections import defaultdict
from typing import DefaultDict, List, Dict, Tuple, Sequence, Union, Iterable

__all__ = ["ReciprocalRankFusion"]


class ReciprocalRankFusion:
    """Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    k : int, default 60
        Constant that controls the contribution of each position.  The original
        RRF paper suggests *k ≈ 60* for web search, but you can tune this
        hyper‑parameter on a validation set if desired.
    """

    def __init__(self, k: int = 60) -> None:  # noqa: D401  (simple docstring OK)
        if not isinstance(k, int) or k <= 0:
            raise ValueError("'k' must be a positive integer, got %r" % k)
        self.k = k

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def fuse(
        self,
        rankings: Sequence[Sequence[str]],
        /,
        *,
        top_k: int | None = None,
        return_scores: bool = False,
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """Fuse multiple ranked lists.

        Parameters
        ----------
        rankings : Sequence[Sequence[str]]
            Each element is a ranking (best at index 0).
        top_k : int | None, default ``None``
            If given, truncate the fused result to the *top_n* docs.
        return_scores : bool, default *False*
            Return ``(doc_id, score)`` tuples instead of just doc_ids.

        Returns
        -------
        list[str] | list[tuple[str, float]]
            The fused ranking; optionally with scores if *return_scores* is set.
        """
        if not rankings:
            return []

        # Accumulate RRF scores.
        scores: DefaultDict[str, float] = defaultdict(float)
        for rank in rankings:
            for position, doc_id in enumerate(rank):
                # +1 because positions are zero‑based in Python but 1‑based in RRF.
                scores[doc_id] += 1.0 / (self.k + position + 1)

        # Sort by score (desc) then doc_id (asc) to ensure deterministic order.
        sorted_items: List[Tuple[str, float]] = sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )

        if top_k is not None and top_k > 0:
            sorted_items = sorted_items[:top_k]

        if return_scores:
            return sorted_items
        return [doc_id for doc_id, _ in sorted_items]

    def score_dict(
        self, rankings: Sequence[Sequence[str]], /, *, normalize: bool = False
    ) -> Dict[str, float]:
        """Return *all* RRF scores as a mapping ``{doc_id: score}``.

        The *normalize* flag rescales all scores to the *[0, 1]* range so that
        they are easier to compare with other scoring functions.
        """
        items = self.fuse(rankings, return_scores=True)  # type: ignore[arg-type]
        if not items:
            return {}

        scores = dict(items)
        if normalize:
            max_score = next(iter(scores.values()))  # first item has max score
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}
        return scores

    def __call__(
        self,
        rankings: Sequence[Sequence[str]],
        /,
        *,
        top_n: int | None = None,
    ) -> List[str]:
        """Call shorthand for :py:meth:`fuse`.  Returns only doc_ids."""
        return self.fuse(rankings, top_k=top_n, return_scores=False)
