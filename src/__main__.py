import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

load_dotenv()

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
        from src.retrieval import lexical

        index = lexical.build_index(Path(data_directory), max_chunk_size)
        target = lexical.save_index(index, Path(save_directory))
        print(
            f"Indexed {index.doc_count} chunks "
            f"({len(index.scorer.vocab_dict)} terms, "
            f"avgdl {index.avgdl:.0f}) "
            f"-> {target}"
        )

    def search(
        self,
        query: str,
        k: int = 5,
        processed_directory: str = "data/processed",
    ) -> None:
        """
            print the top-k BM25 results for a single query
        """
        from src.retrieval import lexical

        index = lexical.load_index(Path(processed_directory))
        ranked = lexical.search(index, str(query), int(k))
        if not ranked:
            print("no results")
            return
        for rank, (chunk_id, score) in enumerate(ranked, start=1):
            file_path, first, last, _ = index.chunks[chunk_id]
            print(
                f"{rank}. {file_path} "
                f"[{first}:{last}] score={score:.2f}"
            )
    
    def answer(
        self,
        query,
        k: int = 5,
        data_directory: str = "data/raw",
        processed_directory: str = "data/processed",
        hops: int = 2,
        max_new_tokens: int = 512,
    ) -> None:
    """
        retrieve -> graph expand -> answer a query
    """
    from src.extraction import extractor
    from src.graph import neo4j_client
    from src.pipeline import query_pipeline
    from src.retrieval import lexical

    index = lexical.load_index(Path(processed_directory))
    driver = neo4j_client.get_driver()
    model = extractor.load_model()

    result = query_pipeline.answer_query(
        str(query), index, driver, Path(data_directory), model,
        k=int(k), hops=int(hops), max_new_tokens=int(max_new_tokens)
    )

    print("Sources:")
    for file_path, first, last, in result["sources"]
        print(f"  {file_path} [{first}:{last}]")
    print()
    print("Answer:")
    print(result["answer"])

    driver.close()


def main() -> None:
    """Run the Fire CLI, converting any unhandled error into a message."""
    try:
        fire.Fire(RagCLI)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001 — CLI boundary, never traceback
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

