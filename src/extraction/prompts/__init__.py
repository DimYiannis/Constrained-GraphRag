from src.chunking.spans import Chunk

from .code_prompt import build_prompt as _build_code_prompt
from .text_prompt import build_prompt as _build_text_prompt


def build_prompt(chunk: Chunk) -> str:
    """
        route a chunk to its extraction prompt by source_type.
    """
    if chunk.source_type == "code":
        return _build_code_prompt(chunk.file_path, chunk.text)
    return _build_text_prompt(chunk.file_path, chunk.text)
