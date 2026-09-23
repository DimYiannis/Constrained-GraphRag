"""
recall@k eval for lexical / semantic / hybrid retrieval against
evaluation/test_queries.json. A question is a hit if any retrieved chunk
covers >= COVERAGE of a ground-truth span.

Arms per retrieval mode, all scored on the same questions:
    {mode}           top-k seeds
    {mode}+graph     top-k seeds + up to expand_budget graph-expanded chunks
    {mode}@matched   same retriever, top-N where N = the +graph arm's chunk
                     count for that question -> the fair baseline for +graph
                     ("does the graph beat just retrieving more?")
    {mode}+random    top-k seeds + as many chunks as +graph added, drawn at
                     random from the graph's Chunk nodes -> leakage control:
                     if the graph only holds ground-truth chunks, random
                     picks from it score hits too
"""

import json
import random
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from src.graph import neo4j_client, traversal
from src.retrieval import lexical
from src.retrieval import fusion
from src.retrieval.semantic import embeddings

load_dotenv()

DEFAULT_TEST_QUERIES = Path("evaluation/test_queries.json")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_K_VALUES = (3, 5, 10)
DEFAULT_HOPS = 2
DEFAULT_MODES = ("lexical",)
COVERAGE = 0.5
GRAPH_ARMS = ("+graph", "@matched", "+random")


def _overlaps(truth_first: int, truth_last: int, b_first: int, b_last: int) -> bool:
    """
    true if two spans overlap, requiring at least COVERAGE of the
    ground-truth span to actually be covered - a single-character
    graze doesn't count as a hit
    """
    intersection = min(truth_last, b_last) - max(truth_first, b_first)
    if (intersection / (truth_last - truth_first)) < COVERAGE:
        return False
    return truth_first < b_last and truth_last > b_first


def _is_hit(sources: list[dict], chunks: list[tuple[str, int, int]]) -> bool:
    """
    true if a ground-truth source overlaps any retrieved chunk
    """
    for source in sources:
        for file_path, first, last in chunks:
            if file_path != source["file_path"]:
                continue
            if _overlaps(
                source["first_character_index"],
                source["last_character_index"],
                first,
                last,
            ):
                return True
    return False


def _seed_chunks(
    mode: str,
    query: str,
    k: int,
    index: lexical.Index,
    semantic_matrix,
    semantic_model,
    candidates: int,
) -> list[tuple[str, int, int]]:
    """
    run one retrieval mode, return (file_path, first, last) seed chunks

    args:
        mode: "lexical", "semantic", or "hybrid"
        query
        k: number of results wanted
        index: bm25 index - always needed, hybrid uses it too
        semantic_matrix: l2-normalized embedding matrix, or None if
            mode == "lexical" (never loaded in that case)
        semantic_model: loaded SentenceTransformer, or None
        candidates: hybrid's per-retriever candidate pool size before
            fusing down to k (see fusion.FUSION_CANDIDATES)
    """
    if mode == "lexical":
        ranked = lexical.search(index, query, k)
    elif mode == "semantic":
        ranked = embeddings.semantic_top_k(semantic_matrix, semantic_model, query, k)
    elif mode == "hybrid":
        ranked = fusion.hybrid_top_k(
            index, semantic_matrix, semantic_model, query, k, candidates=candidates
        )
    else:
        raise ValueError(f"unknown mode {mode!r} (expected lexical/semantic/hybrid)")

    return [
        (index.chunks[cid][0], index.chunks[cid][1], index.chunks[cid][2])
        for cid, _ in ranked
    ]


def _graph_pool(driver, database: str | None) -> list[tuple[str, int, int]]:
    """every Chunk node in the graph, sorted so random draws are reproducible"""
    records, _, _ = neo4j_client.run_query(
        driver,
        "MATCH (c:Chunk) RETURN c.file_path AS file_path, c.first AS first, c.last AS last",
        database=database,
    )
    return sorted((r["file_path"], r["first"], r["last"]) for r in records)


def evaluate(
    test_queries_path: Path = DEFAULT_TEST_QUERIES,
    processed_directory: Path = DEFAULT_PROCESSED_DIR,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    modes: tuple[str, ...] = DEFAULT_MODES,
    hops: int = DEFAULT_HOPS,
    use_graph: bool = True,
    candidates: int = fusion.FUSION_CANDIDATES,
    database: str | None = None,
    show_progress: bool = True,
    expand_budget: int | None = None,
):
    """
    recall@k for one or more retrieval modes, each optionally with the
    graph arms described in the module docstring

    a generator, not a function returning a dict all at once - yields
    (k, entry) as each k value finishes, so a caller (main(), below)
    can print/display results progressively instead of holding
    everything until every k and every mode is done. The corpus is
    loaded once regardless - this doesn't cost anything beyond a
    different way of handing results back.

    args:
        test_queries_path
        processed_directory: dir holding the saved bm25 index +
            embeddings matrix
        k_values: bm25/semantic/hybrid top-k values
        modes: which retrieval modes to evaluate - any of
            "lexical", "semantic", "hybrid"
        hops: graph expansion depth
        use_graph: also compute each mode's graph arms
        candidates: hybrid's per-retriever candidate pool before fusion
        database: neo4j database
        show_progress: tqdm bar over queries, one per k value
        expand_budget: max graph-expanded chunks added per question;
            None -> k, so +graph uses at most 2k chunks

    yields:
        (k, {mode: recall, f"{mode}+graph": recall,
             f"{mode}@matched": recall, f"{mode}+random": recall,
             f"{mode}+graph_added": mean expanded chunks per question,
             "per_split": {split: {arm: recall, ..., "n": count}}})
        one pair per k value, in k_values order
    """
    queries = json.loads(Path(test_queries_path).read_text())
    index = lexical.load_index(Path(processed_directory))
    driver = neo4j_client.get_driver() if use_graph else None
    pool = _graph_pool(driver, database) if use_graph else []

    # only load the embedding model/matrix if a mode actually needs them -
    # keeps a pure lexical run fast and dependency-light
    needs_semantic = any(mode in ("semantic", "hybrid") for mode in modes)
    semantic_matrix = (
        embeddings.load_embeddings(processed_directory) if needs_semantic else None
    )
    semantic_model = embeddings.load_model() if needs_semantic else None

    arms = [
        f"{mode}{suffix}"
        for mode in modes
        for suffix in ("", *(GRAPH_ARMS if use_graph else ()))
    ]

    try:
        for k in k_values:
            budget = k if expand_budget is None else expand_budget
            hits = {arm: 0 for arm in arms}
            added = {mode: 0 for mode in modes}
            # split -> {arm: hits, ..., "n": count}
            split_hits: dict[str, dict] = {}

            iterator = tqdm(
                queries,
                desc=f"recall@{k} ({', '.join(modes)})",
                unit="query",
                disable=not show_progress,
            )
            for qi, query in enumerate(iterator):
                split = query["split"]
                if split not in split_hits:
                    split_hits[split] = {arm: 0 for arm in arms}
                    split_hits[split]["n"] = 0
                split_hits[split]["n"] += 1

                for mode in modes:
                    # retrieve enough for the @matched arm once; every
                    # retriever's top-n is a prefix of its top-(n+m)
                    ranked = _seed_chunks(
                        mode,
                        query["question"],
                        k + budget if use_graph else k,
                        index,
                        semantic_matrix,
                        semantic_model,
                        candidates,
                    )
                    seed_chunks = ranked[:k]
                    retrieved = {mode: seed_chunks}

                    if use_graph:
                        expanded = traversal.expand_chunks(
                            driver,
                            seed_chunks,
                            hops=hops,
                            database=database,
                            max_expanded=budget,
                        )
                        expanded_chunks = [
                            (chunk["file_path"], chunk["first"], chunk["last"])
                            for chunk in expanded
                        ]
                        n_added = len(expanded_chunks)
                        added[mode] += n_added

                        seed_set = set(seed_chunks)
                        candidates_random = [c for c in pool if c not in seed_set]
                        rng = random.Random(f"{k}:{qi}:{mode}")
                        random_chunks = rng.sample(
                            candidates_random, min(n_added, len(candidates_random))
                        )

                        retrieved[f"{mode}+graph"] = seed_chunks + expanded_chunks
                        retrieved[f"{mode}@matched"] = ranked[: k + n_added]
                        retrieved[f"{mode}+random"] = seed_chunks + random_chunks

                    for arm, chunks in retrieved.items():
                        if _is_hit(query["sources"], chunks):
                            hits[arm] += 1
                            split_hits[split][arm] += 1

            total = len(queries)
            entry: dict = {arm: hits[arm] / total for arm in arms}
            if use_graph:
                for mode in modes:
                    entry[f"{mode}+graph_added"] = added[mode] / total
            entry["per_split"] = {
                split: {
                    **{arm: counts[arm] / counts["n"] for arm in arms},
                    "n": counts["n"],
                }
                for split, counts in split_hits.items()
            }
            yield k, entry
    finally:
        if driver is not None:
            driver.close()


def main(
    test_queries_path: str = str(DEFAULT_TEST_QUERIES),
    processed_directory: str = str(DEFAULT_PROCESSED_DIR),
    k: str = "3,5,10",
    modes: str = "lexical",
    hops: int = DEFAULT_HOPS,
    use_graph: bool = True,
    candidates: int = fusion.FUSION_CANDIDATES,
    database: str | None = None,
    show_progress: bool = True,
    reveal_delay: float = 0.0,
    expand_budget: int | None = None,
) -> None:
    """
    CLI entry: print a recall@k table for the requested mode(s),
    one k value at a time as each finishes (evaluate() is a
    generator - see its docstring)

    reveal_delay: seconds to pause after printing each k value's
        block, before moving on to the next. 0 by default (no
        reason to slow down a real evaluation run) - set to
        something like 2-3 when recording a demo, so a viewer
        actually has time to read one block before the next appears.
    """
    # Fire auto-splits a comma-containing CLI arg into a tuple before this
    # function even runs, so k/modes may already be a tuple, or a plain
    # str/int if there was only one value, handle all three shapes.
    k_values = tuple(int(x) for x in k) if isinstance(k, tuple) else (int(k),)
    mode_values = (
        tuple(str(m).strip() for m in modes)
        if isinstance(modes, tuple)
        else (str(modes).strip(),)
    )
    suffixes = ("", *(GRAPH_ARMS if use_graph else ()))

    for kk, entry in evaluate(
        Path(test_queries_path),
        Path(processed_directory),
        k_values,
        mode_values,
        int(hops),
        bool(use_graph),
        int(candidates),
        database,
        bool(show_progress),
        None if expand_budget is None else int(expand_budget),
    ):
        print(f"--- recall@{kk} ---")
        for mode in mode_values:
            line = "  " + "  ".join(
                f"{mode}{s}: {entry[f'{mode}{s}']:.3f}" for s in suffixes
            )
            if use_graph:
                line += f"  (graph added {entry[f'{mode}+graph_added']:.1f} chunks/question)"
            print(line)
        for split, s in entry["per_split"].items():
            parts = [
                f"{mode}{suffix}={s[f'{mode}{suffix}']:.3f}"
                for mode in mode_values
                for suffix in suffixes
            ]
            print(f"    {split} (n={s['n']}): " + " ".join(parts))
        if reveal_delay > 0:
            time.sleep(reveal_delay)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
