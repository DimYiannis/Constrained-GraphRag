from pathlib import Path

from src.cache import cache
from src.chunking.chunk_corpus import read_text
from src.extraction import extractor
from src.graph import traversal
from src.retrieval import lexical

DEFAULT_K = 5
DEFAULT_HOPS = 2
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_CACHE_DIR = Path("data/cache")

PROMPT_TEMPLATE= """\
Answer the question using only the context below. if the context\
doesnt contain the answer, say so - do not make anything up.

Context:
{context}

Question: {query}
Answer:"""

def _reslice(data_dir: Path, file_path: str, first: int, last: int) -> str:
    """
        re-slice a chunk's text from the source file
    """
    text = read_text(data_dir / file_path) # join into one path
    return text[first:last] if text is not None else ""

def answer_query(
    query: str,
    index,
    driver,
    data_dir: Path,
    model,
    k: int = DEFAULT_K,
    hops: int = DEFAULT_HOPS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    show_progress: bool = True,
) -> dict:
    """
        retrieve -> graph expand -> prompt -> answer

        args:
            query
            index: bm25 index
            dirver: Neo4j driver
            data_dir: corpus root
            model: model from extractor
            k: bm25 top-k
            hops: graph expansion depth
            max_new_tokens
            cache_dir: exact-match query cache location
            show_progress: print a one-line status per stage (retrieval,
                graph expansion, generation) as it happens, instead of
                silence until the final result

        return:
            {"answer": str, "sources": [(file_path, first, last, origin),...]}
            origin is "lexical" (BM25 seed) or "graph" (graph-expanded),
            lets a caller show which chunks the graph actually contributed.

        an identical (query, k, hops, max_new_tokens) call is served from
        `cache_dir` without re-running retrieval/graph-expand/generation,
        the model still has to be loaded by the caller either way, this
        only skips the actual per-query work.
    """
    def status(message: str) -> None:
        if show_progress:
            print(message)

    query_cache = cache.load_cache(cache_dir)
    cached = cache.get_cached_result(query_cache, query, k, hops, max_new_tokens)
    if cached is not None:
        status("[cache] hit - skipping retrieval, graph expansion, and generation")
        return cached

    status("[1/3] retrieving...")
    ranked = lexical.search(index, query, k)
    seed_chunks = [
        (index.chunks[cid][0], index.chunks[cid][1], index.chunks[cid][2], "lexical")
        for cid, _ in ranked
    ]
    status(f"[1/3] retrieved {len(seed_chunks)} seed chunks")

    status(f"[2/3] expanding graph ({hops} hops)...")
    expand_input = [(fp, first, last) for fp, first, last, _origin in seed_chunks]
    expanded = traversal.expand_chunks(driver, expand_input, hops=hops)
    expanded_chunks = [
        (chunk["file_path"], chunk["first"], chunk["last"], "graph")
        for chunk in expanded
    ]
    status(f"[2/3] graph expansion added {len(expanded_chunks)} new chunks")

    all_chunks = seed_chunks + expanded_chunks

    context = "\n---\n".join(
        _reslice(data_dir, file_path, first, last)
        for file_path, first, last, _origin in all_chunks
    )

    status(f"[3/3] generating answer (up to {max_new_tokens} tokens)...")
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)
    answer = model(prompt, max_new_tokens=max_new_tokens)
    status("[3/3] done")

    result = {"answer": answer, "sources": all_chunks}
    cache.store_result(query_cache, query, k, hops, max_new_tokens, result)
    cache.save_cache(query_cache, cache_dir)
    return result
