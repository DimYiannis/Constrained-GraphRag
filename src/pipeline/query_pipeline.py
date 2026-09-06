from pathlib import Path

from src.chunking.chunk_corpus import read_text
from src.extraction import extractor
from src.graph import traversal
from src.retrieval import lexical

DEFAULT_K = 5
DEFAULT_HOPS = 2
DEFAULT_MAX_NEW_TOKENS = 512

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
        
        return:
            {"answer": str, "sources": [(file_path, first, last),...]}
            sources includes both the BM25 seeds and graph-expanded chunks 
    """
    print("Debug query=", repr(query), "k=", k, "index.doc_coun=", index.doc_count)
    
    ranked = lexical.search(index, query, k)
    print("Debug ranked=", ranked)
    
    seed_chunks = [
        (index.chunks[cid][0], index.chunks[cid][1], index.chunks[cid][2])
        for cid, _ in ranked
    ]
    print("Debug seed_chunks=", seed_chunks)

    expanded = traversal.expand_chunks(driver, seed_chunks, hops=hops)
    expanded_chunks = [(chunk["file_path"], chunk["first"], chunk["last"]) for chunk in expanded]

    all_chunks = seed_chunks + expanded_chunks
    print("Debug all chunks=", all_chunks)

    context = "\n---\n".join(
        _reslice(data_dir, file_path, first, last)
        for file_path, first, last in all_chunks
    )

    prompt = PROMPT_TEMPLATE.format(context=context, query=query)
    answer = model(prompt, max_new_tokens=max_new_tokens)

    return {"answer": answer, "sources": all_chunks}
