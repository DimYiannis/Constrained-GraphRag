"""
shared top-k selection for every retriever over a full score array
"""

import numpy as np


def top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    """
    the k highest-scoring chunk ids, score descending, ties on the
    lower chunk id - across the whole array, not just inside a
    partial selection, so top_k(s, n) is always a prefix of
    top_k(s, n + m)

    args:
        scores: one score per chunk_id
        k: number of results wanted; k <= 0 yields no results

    return:
        (chunk_id, score) pairs
    """
    wanted = min(k, scores.shape[0])
    if wanted <= 0:
        return []
    # partial select, then pull in every chunk tied with the k-th score so
    # the id tie-break sees all of them, not whichever argpartition kept
    selected = np.argpartition(-scores, wanted - 1)[
        :wanted
    ]  # np.argpartition is a partial sort
    boundary = scores[selected].min()  # the k-th best score
    candidates = np.flatnonzero(
        scores >= boundary
    )  # ids of every chunk scoring at least that, ties included
    order = np.lexsort((candidates, -scores[candidates]))[
        :wanted
    ]  # sort by score desc, then id asc (lexsort: last key is primary)
    return [(int(candidates[i]), float(scores[candidates[i]])) for i in order]
