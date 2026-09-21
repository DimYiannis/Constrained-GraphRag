"""
    Reciprocal Rank Fusion of lexical + semantic rankings. 
"""

from src.retrieval import lexical
from src.retrieval.semantic.embeddings import semantic_top_k

RRF_C = 60 # Cormack et al.'s standard RRF constant.

FUSION_CANDIDATES = 100

def fuse(
    lexical_ranked: list[tuple[int, float]],
    semantic_ranked: list[tuple[int, float]],
    k: int,
    c: int = RRF_C
) -> list[tuple[int, float]]:
    """
        Reciprocal Rank Fusion of a lexical and a semantic ranking.

        Why Use RRF Over Score Addition:
            Incompatible Scales -> Lexical scores (BM25) and semantic scores (cosine similarity or dot product) 
                use completely different measurement scales
            No Normalization Needed
            Robust Performance
    """
    if k <= 0:
        return []
    rrf_scores: dict[int, float] = {}
    for ranked in (lexical_ranked, semantic_ranked):
        for rank, (chunk_id, _) in enumerate(ranked, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (c + rank)
    fused = sorted(rrf_scores.items(), key=lambda item: (-item[1], item[0]))
    return fused[:k]


def hybrid_top_k(
    index: lexical.Index,
    embeddings,
    model,
    query,
    k,
    c: int = RRF_C,
    candidates: int = FUSION_CANDIDATES
) -> list[tuple[int, float]]:
    """
        lexical + semantic search fused by RRF

        args:
            index: bm25 index
            embeddings: l2-normalized embedding matrix
            model: loaded SentenceTransformer
            query
            k: num of results
            candidates: how many results to pull from
                each retrieval
            c: RRF constant
        
        return:
            (chunk_id, rrf_score)
    """
    lexical_rank = lexical.search(index, query, candidates)
    semantic_rank = semantic_top_k(embeddings, model, query, candidates)
    return fuse(lexical_rank, semantic_rank, k, c=c)
