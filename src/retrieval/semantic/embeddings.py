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
from src.chunnking.chunk_corpus import read_text

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_FILENAME = "embeddings.py"

def load_model(model: str = MODEL_NAME) -> SentenceTransformer:
    model = SentenceTransformer(model, device="cpu")
    return model

def _get_chunks(index: Index, data_dir: Path, show_progress: bool = True) -> list[str]:
    """
        reslice and build chunk_id and spans

        args:
            index: chunk metadata
            data_dir: corpus root, chunk file_paths are relative to this
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
            text = read_text(data_dir / file_path)
            cache[file_path] = text if text is not None else ""
        chunk_text = cache[file_path][first:last]
        texts.append(f"{file_path}\n{chunk_text}")
    return texts

def build_embeddings(
    index: Index,
    data_dir: Path,
    model: SentnenceTransformer,
    batch_size: int = 64,
    show_progress: bool = True
) -> np.ndarray:
    """
        embed every chunk in the index

        return l2-normalised matrix, position = chunk_id
    """
    if model is None:
        load_model()
    texts = _get_chunks(index, data_dir, show_progress=show_progress)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)

def save_embeddings(matrix: np.ndarray, save_dir: Path) -> Path:
    """
        save embeddings matrix next to the lexical index

        return path of the written .npy file
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    target = save_dir / EMBEDDING_FILENAME
    np.save(target, matrix)
    return target

def load_embeddings(save_dir: Path) -> np.ndarray:
    target = save_dir / EMBEDDING_FILENAME
    if not target.is_file():
        raise FileNotFoundError(f"No embeddings at {target}")
    try:
        return cast(np.ndarray, np.load(target))
    except (OSError, ValueError) as exc:
        raise ValueError(f"corrupt embeddings file {target}: {exc}") from exc

