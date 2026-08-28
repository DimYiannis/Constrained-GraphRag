"""
    inverted-index construction, persistence, and BM25 search.

    - chunking is done here directly from a corpus dir (self-contained,
        matching the old project's design) rather than accepting an
        already-chunked list.
    - chunk text is deliberately NOT stored in the persisted index:
        consumers re-slice the source file using (file_path, first, last)
        to get identical spans back, keeping the index file small.
"""
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import bm25s
from tqdm import tqdm

from src.chunking.chunk_corpus import chunk, iter_corpus_files, read_text

from .tokenizer import strip_stopwords, tokenize

ChunkMeta = tuple[str, int, int, int]

INDEX_FILENAME = "index.pkl"
SCORER_DIRNAME = "bm25s"
# bumped when the on-disk layout changes.
FORMAT_VERSION = 1

# BM25 params -> tuned for a code+text corpus, baked in at index time.
DEFAULT_K1 = 1.3
DEFAULT_B = 0.85


@dataclass
class Index:
    """
        BM25 index over the chunked corpus

        attrs:
            max_chunk_size
            chunks: per-chunk metadata; position = chunk id, and the
                same position bm25s returns from a query
            scorer: the bm25s retriever, built from our token lists
            avgdl: average chunk length in tokens (reported stat;
                bm25s does its own length normalization internally)
    """

    max_chunk_size: int
    chunks: list[ChunkMeta]
    scorer: bm25s.BM25
    avgdl: float

    @property
    def doc_count(self) -> int:
        """
            number of chunks in the index
        """
        return len(self.chunks)


def build_index(
    data_dir: Path,
    max_chunk_size: int = 2000,
    show_progress: bool = True,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> Index:
    """
        chunk and tokenize the corpus, then hand the token lists to bm25s

        args:
            data_dir: corpus root
            max_chunk_size
            show_progress: display tqdm bar
            k1: bm25 term-frequency saturation (index-time under bm25s)
            b: bm25 length normalization (index-time under bm25s)

        return:
            the in-memory index
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be a positive integer")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"corpus dir not found: {data_dir}")
    files = iter_corpus_files(data_dir)
    if not files:
        raise ValueError(f"no indexable files under {data_dir}")

    chunks: list[ChunkMeta] = []
    corpus_tokens: list[list[str]] = []
    total_tokens = 0

    iterator = tqdm(files, desc="indexing", unit="files",
                    disable=not show_progress)
    for path in iterator:
        text = read_text(path)
        if text is None:
            continue
        rel_path = Path(os.path.relpath(path)).as_posix()
        # path tokens let a chunk match questions that name its file
        # (e.g. lora.py) without quoting any code content.
        path_tokens = tokenize(rel_path.removeprefix(f"{data_dir}/"))
        for piece in chunk(text, rel_path, max_chunk_size):
            tokens = tokenize(piece.text) + path_tokens
            if not tokens:
                continue
            chunks.append(
                (piece.file_path, piece.first, piece.last, len(tokens))
            )
            corpus_tokens.append(tokens)
            total_tokens += len(tokens)

    if not chunks:
        raise ValueError("corpus produced no chunks")

    # bm25s takes the token lists directly, so our tokenizer (subtokens
    # + path tokens) stays the thing that decides what is matchable.
    scorer = bm25s.BM25(k1=k1, b=b, method="lucene")
    scorer.index(corpus_tokens, show_progress=show_progress)

    return Index(
        max_chunk_size=max_chunk_size,
        chunks=chunks,
        scorer=scorer,
        avgdl=total_tokens / len(chunks),
    )


def search(index: Index, query: str, k: int) -> list[tuple[int, float]]:
    """
        return the k best chunks for a query, best first.

        args:
            index: chunk metadata + bm25s scorer
            query: free-text query, tokenized identically to chunks
            k: number of results wanted; k <= 0 yields no results

        return:
            (chunk_id, score) pairs, score descending; ties break on the
            lower chunk_id so results are deterministic. Empty for empty
            or fully out-of-vocabulary queries.
    """
    if k <= 0:
        return []
    terms = list(dict.fromkeys(tokenize(query)))
    if not terms:
        return []
    # Stopwords are dropped from the query but kept in the index: removing
    # them from the index would change every chunk length and every idf.
    terms = strip_stopwords(terms)
    # bm25s errors if asked for more documents than it holds.
    wanted = min(k, index.doc_count)
    ids, scores = index.scorer.retrieve([terms], k=wanted, show_progress=False)
    ranked = [
        (int(chunk_id), float(score))
        for chunk_id, score in zip(ids[0], scores[0])
        if score > 0
    ]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked


def save_index(index: Index, save_dir: Path) -> Path:
    """
        save chunk metadata as a dict using pickle - consumers reslice
        chunks based on these offsets. the bm25s scorer is written next
        to it in its own format.

        args:
            index
            save_dir: target dir
        return:
            path of the written index file
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    target = save_dir / INDEX_FILENAME
    payload = {
        "version": FORMAT_VERSION,
        "max_chunk_size": index.max_chunk_size,
        "chunks": index.chunks,
        "avgdl": index.avgdl,
    }
    with open(target, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    index.scorer.save(str(save_dir / SCORER_DIRNAME), show_progress=False)
    return target


def load_index(processed_dir: Path) -> Index:
    """
        load a previously saved index

        args:
            processed_dir

        return:
            the loaded index
    """
    target = processed_dir / INDEX_FILENAME
    if not target.is_file():
        raise FileNotFoundError(f"no index at {target}")
    try:
        with open(target, "rb") as handle:
            payload = pickle.load(handle)
        if payload["version"] != FORMAT_VERSION:
            raise ValueError(
                f"index format {payload['version']} unsupported - rebuild"
            )
        scorer = bm25s.BM25.load(
            str(processed_dir / SCORER_DIRNAME), mmap=True, show_progress=False
        )
        return Index(
            max_chunk_size=payload["max_chunk_size"],
            chunks=payload["chunks"],
            scorer=scorer,
            avgdl=payload["avgdl"],
        )
    except (pickle.UnpicklingError, KeyError, EOFError, OSError) as exc:
        raise ValueError(f"corrupt index file {target}: {exc}") from exc
