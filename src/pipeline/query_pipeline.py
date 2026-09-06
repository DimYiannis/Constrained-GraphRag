from pathlib import Path

from src.chunking.chunk_corpus import read_text
from src.extraction import extractor
from src.graph import traversal
from src.retrieval import lexical

DEFAULT_K = 5
DEFAULT_HOPS = 2
DEFAULT_MAX_NEW_TOKENS = 512

PROMPT_TEMPLATE= """`\
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



