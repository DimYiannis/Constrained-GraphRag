"""
    recall@k eval for the lexical retrieval pipeline
    against evaluation/test_queries.json
    Formula:
                                Number of Relevant Items in Top K
        Recall@K=       ------------------------------------------------------
                        Total Number of Relevant Items in Ground Truth Dataset
"""

import json
from pathlib import Path

from dotenv import load_dotenv

from src.graph import neo4j_client, traversal
from src.retrieval import lexical

load_dotenv()

DEFAULT_TEST_QUERIES = Path("evaluation/test_queries.json")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_K_VALUES = (3, 5, 10)
DEFAULT_HOPS = 2
COVERAGE = 0.5

def _overlaps(truth_first: int, truth_last: int, b_first: int, b_last: int) -> bool:
    """
        true if two spans overlap
    """
    intersection = min(truth_last, b_last) - max(truth_first, b_first)
    if (intersection / (truth_last - truth_first)) < COVERAGE:
        return False
    return truth_first < b_last and truth_last > b_first

def _is_hit(sources: list[dict], chunks: list[tuple[str, int, int]]) -> bool:
    """
        true if groound-truth source overlaps any retrieved chunk
    """
    for source in sources:
        for file_path, first, last in chunks:
            if file_path != source["file_path"]:
                continue
            if _overlaps(
                source["first_character_index"], source["last_character_index"],
                first, last,
            ):
                return True
    return False

def evaluate(
    test_queries_path: Path = DEFAULT_TEST_QUERIES,
    processed_directory: Path = DEFAULT_PROCESSED_DIR,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    hops: int = DEFAULT_HOPS,
    use_graph: bool = True,
    database: str | None = None,
) -> dict:
    """
        recall@k for lexical-only vs lexical_graph

        args:
            test_queries_path
            processed_directory: dir holding the saved bm25 index
            k_values:: bm25 top k values
            hops
            use_graph: + compute lexical_graph recall
            database: neo4j database
        
        return:
            k: "lexical": recall, "lexical+graph": recall,
            "per_split": {
                    "docs": {"lexical": 0.8, "lexical+graph": 0.8, "n": 100},
                    "code": {"lexical": 0.7, "lexical+graph": 0.7, "n": 99},                                         
                }
    """
    queries = json.loads(Path(test_queries_path).read_text())
    index = lexical.load_index(Path(processed_directory))
    driver = neo4j_client.get_driver() if use_graph else None

    result = {}
    try:
        for k in k_values:
            lexical_hits = 0
            graph_hits = 0
            split_hits: dict[str, list[int]] = {} # split -> [lexical, graph, total]

            for query in queries:
                split = query["split"]
                split_hits.setdefault(split, [0, 0, 0,])
                split_hits[split][2] += 1

                ranked = lexical.search(index, query["question"], k)
                seed_chunks = [
                    (index.chunks[cid][0], index.chunks[cid][1], index.chunks[cid][2])
                    for cid, _ in ranked
                ]

                if _is_hit(query["sources"], seed_chunks):
                    lexical_hits += 1
                    split_hits[split][0] += 1

                if use_graph:
                    expanded = traversal.expand_chunks(
                        driver, seed_chunks, hops=hops, database=database
                    )
                    expanded_chunks = [
                        (chunk["file_path"], chunk["first"], chunk["last"]) for chunk in expanded
                    ]
                    all_chunks = seed_chunks + expanded_chunks
                    if _is_hit(query["sources"], all_chunks):
                        graph_hits += 1
                        split_hits[split][1] += 1

            total = len(queries)
            entry = {"lexical": lexical_hits / total}
            if use_graph:
                entry["lexical+graph"] = graph_hits / total
            entry["per_split"] = {
                split: {
                    "lexical": counts[0] / counts[2],
                    **(
                        {"lexical+graph": counts[1] / counts[2]}
                        if use_graph else {}
                    ),
                    "n": counts[2],
                }
                for split, counts in split_hits.items()
            }
            result[k] = entry
    finally:
        if driver is not None:
            driver.close()
    
    return result


def main(
    test_queries_path: str = str(DEFAULT_TEST_QUERIES),
    processed_directory: str = str(DEFAULT_PROCESSED_DIR),
    k: str = "3,5,10",
    hops: int = DEFAULT_HOPS,
    use_graph: bool = True,
    database: str | None = None,
) -> None:
    results = evaluate(
        Path(test_queries_path), Path(processed_directory),
        k, int(hops), bool(use_graph), database,
    )

    print(results)


if __name__ == "__main__":
    import fire

    fire.Fire(main)