"""
    offline pipeline: corpus -> chunk -> extract -> load into Neo4j
    whole corpus is approx 28k chunks, extraction would take long
    so limit defaults to a smaller number so that the extraction
    finishes in mins.
"""

import os
from pathlib import Path

from src.chunking.chunk_corpus import iter_corpus_files, read_text, chunk
from src.extraction import extractor
from src.graph import loader, neo4j_client

DEFAULT_LIMIT = 20
DEFAULT_MAX_CHUNK_SIZE = 2000

def run(
    data_dir: Path,
    limit: int | None = DEFAULT_LIMIT,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,   
    max_new_tokens: int = extractor.DEFAULT_MAX_NEW_TOKENS,
    database: str | None = None,
) -> int:
    """
        chunk every file under data_dir, extract entities/relationships
        from each chunk, and load them into Neo4j

        return the number of chunks processed
    """
    driver = neo4j_client.get_driver()
    model = extractor.load_model()
    generator = extractor.build_generator(model)

    processed = 0
    try:
        for path in iter_corpus_files(data_dir):
            if limit is not None and processed >= limit:
                break
            text = read_text(path)
            if text is None:
                continue
            rel_path = Path(os.path.relpath(path, data_dir)).as_posix()
            for piece in chunk(text, rel_path, max_chunk_size):
                if limit is not None and processed>= limit:
                    break
                print("extracting... \n")
                result = extractor.extract(generator, piece, max_new_tokens=max_new_tokens)
                print({result})
                print()
                print("loading... ")
                loader.load_chunk(driver, piece, result, database=database)
                print("loaded in db")
                processed += 1
    finally:
        driver.close()

    return processed