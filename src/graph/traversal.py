from src.graph.neo4j_client import run_query

# entity-to-entity relation types to traverse.
# MENTIONED_IN is deliberately excluded here -
# used separately to get in/out of the graph.
TRAVERSAL_RELATIONS = "CALLS|IMPORTS|INHERITS_FROM|DEFINED_IN|RELATES_TO|REFERENCES"

DEFAULT_MAX_EXPANDED = 50


def expand_chunks(
    driver,
    chunks: list[tuple[str, int, int]],
    hops: int = 2,
    database: str | None = None,
    max_expanded: int = DEFAULT_MAX_EXPANDED,
) -> list[dict]:
    """
        from the retriever we get the seed chunks
        from there we traverse through shared entities
        to find related chunks that didn't lexically match

        distance 0 = the new chunk mentions the same entity as a seed
        (the cross-chunk link loader.py's MERGE creates), distance n =
        reached through n entity-to-entity edges.

        args:
            driver
            chunks: seed chunks (file path, first, last)
            hops: max entity-to-entity edges to follow
            database
            max_expanded: cap on newly-reached chunks a hub entity
                (high fanout, something referenced everywhere) can
                otherwise blow the expansion up to hundreds of chunks,
                which balloons the prompt and tanks generation time for
                no retrieval benefit

        return:
            [{file_path, first, last, distance, support}, ...] for newly
            reached chunks, best first: closest distance, then most seed
            entities reaching it (support), then file_path/first for a
            deterministic order. seed chunks are excluded before the cap
            is applied, so the cap only counts new chunks
    """
    seeds = [
        {"file_path": fp, "first": first, "last": last}
        for fp, first, last in chunks
    ]
    seed_keys = [[fp, first, last] for fp, first, last in chunks]

    records, _, _ = run_query(
        driver,
        f"""
        UNWIND $chunks AS seed
        MATCH (c:Chunk {{file_path: seed.file_path, first: seed.first, last: seed.last}})
        MATCH (c)<-[:MENTIONED_IN]-(seed_entity)
        MATCH path = (seed_entity)-[:{TRAVERSAL_RELATIONS}*0..{int(hops)}]-(related_entity)
        MATCH (related_entity)-[:MENTIONED_IN]->(expanded:Chunk)
        WHERE NOT [expanded.file_path, expanded.first, expanded.last] IN $seed_keys
        WITH expanded,
             min(length(path)) AS distance,
             count(DISTINCT seed_entity) AS support
        RETURN expanded.file_path AS file_path,
               expanded.first AS first,
               expanded.last AS last,
               distance,
               support
        ORDER BY distance ASC, support DESC, file_path ASC, first ASC
        LIMIT $max_expanded
        """,
        {"chunks": seeds, "seed_keys": seed_keys, "max_expanded": max_expanded},
        database=database,
    )
    return [dict(r) for r in records]
