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
from sentence_transformers import SentenceTransformer

from src.retrieval.lexical import Index
from src.chunking.chunk_corpus import read_text

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_FILENAME = "embeddings.npy"

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
    model: SentenceTransformer | None = None,
    batch_size: int = 64,
    show_progress: bool = True
) -> np.ndarray:
    """
        embed every chunk in the index

        return l2-normalised matrix, position = chunk_id
    """
    if model is None:
        model = load_model()
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
    target = save_dir / EMBEDDINGS_FILENAME
    np.save(target, matrix)
    return target

def load_embeddings(save_dir: Path) -> np.ndarray:
    target = save_dir / EMBEDDINGS_FILENAME
    if not target.is_file():
        raise FileNotFoundError(f"No embeddings at {target}")
    try:
        return cast(np.ndarray, np.load(target))
    except (OSError, ValueError) as exc:
        raise ValueError(f"corrupt embeddings file {target}: {exc}") from exc

def semantic_top_k(
    embeddings: np.ndarray,
    model: SentenceTransformer,
    query: str,
    k: int,
) -> list[tuple[int, float]]:
    """
        return the k best chunks for a query by cosine similarity

        args:
            embeddings: l2 normalised matrix
            model
            query
            k: num of results wanted

        return:
            (chunk_id, score) pairs, score descending, ties break on
            lower chunk_id
    """
    if k <= 0 or embeddings.shape[0] == 0:
        return []
    query_vector = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False,
    )[0].astype(np.float32) # single, L2-normalized, float32, 384-dim vector
    sims = embeddings @ query_vector # cosine similarity colapsed to a dot product
    wanted = min(k, sims.shape[0])
    top_idx = np.argpartition(-sims, wanted - 1)[:wanted] # np.argpartition is a partial sort
    ranked = [(int(i), float(sims[i])) for i in top_idx] # turn raw indices into (chunk_id, score) pairs
    ranked.sort(key=lambda item: (-item[1], item[0])) # sort small wanted-sized list
    return ranked