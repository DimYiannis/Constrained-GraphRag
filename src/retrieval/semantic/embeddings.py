"""
    semantic search with dense retrieval
    relevance is based on cosine similarity amongst
    the vectors generated from the embedding
    model
"""

from pathlib import Path
from typing import cast

from tqdm import tqdm
import numpy as np
from sentence_transformers import SentnenceTransformer

from src.retrieval.lexical import Index

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_FILENAME = "embeddings.py"

def load_model(model: str = MODEL_NAME) -> SentenceTransformer:
    model = SentenceTransformer(model, device="cpu")
    return model

def _get_chunks(index: Index, show_progress: bool = True) -> list[str]:
    """
        reslice and build chunk_id and spans

        args:
            index: chunk metadata
            show_progress: tqdm bar
        
        return:
            chunk texts, position= chunk_id
    """
    cache: dict[str, str] = {}
    texts: list[str] = []
    iterator = tqdm(
        index.chunks, desc="reslicing", unit="chunk",
        disable=not show_progress
    )
    for file_path, first, last, _ in iterator:
        if file_path not in cache:
            with open(file_path, encoding="utf-8") as handle:
                cache[file_path] = handle.read()
        chunk_text = cache[file_path][first:last]
        texts.append(f"{file_path}\n{chunk_text}")
    return texts

def build_embeddings(
    index: Index,
    model: SentnenceTransformer,
    batch_size: int = 64,
    show_progress: bool = True
) -> np.ndarray:
    """
        embed every chunk in the index

        return l2-normalised matrix
    """