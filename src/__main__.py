import sys
from pathlib import Path

import fire

class RagCLI:
    
    def index(
        self,
        max_chunk_size: int = 2000,
        data_directory: str = "data/raw",
        save_directory: str = "data/processed",
    ) -> None:
    """
        chunk the corpus and build the inverted index
    """
    from src import indexer

    print(
        f"Indexed {index.doc_count} chunks "
        f"({len(index.scorer.vocab_dict)} terms, "
        f"avgdl {index.avgdl:.0f}) "
        f" -> {target}"
    )