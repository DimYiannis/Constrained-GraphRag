

from src.graph.neo4j_client import run_query

# entity-to-entity relation types to traverse. 
# MENTIONED_IN is deliberately excluded here -  
# used separately to get in/out of the graph.
TRAVERSAL_RELATIONS = "CALLS|IMPORTS|INHERITS_FROM|DEFINED_IN|RELATES_TO|REFERENCES"

def expand_chunks(
    driver,
    chunks: list[tuple[str, int, int]],
    hops: int = 2,
    database: str | None = None,
) -> list[dict]:
    """
        from bm25 we get the seed chunks
        from there we tranverse through shared entities
        to find related chunks that didnt lexically matched

        args:
            driver
            chunks: seed chunks (file path, first, last)
            hops: max traversal depth
            database

        return:
            [{file_path, first, last}, ...] for newly reached chunks
            seed chunks are excluded from the result
    """

    seeds = [
        {"file_path": fp, "first": first, "last": last}
        for fp, first, last in chunks
    ]

    records, _, _ = run_query(
        driver,
        f"""
        UNWIND $chunks AS seed
        MATCH (c:Chunk {{file_path: seed.file_path, first: seed.first, last: seed.last}})
        MATCH (c)<-[:MENTIONED_IN]-(seed_entity)
        MATCH (seed_entity)-[:{TRAVERSAL_RELATIONS}*1..{hops}]-(related_entity)
        MATCH (related_entity)-[:MENTIONED_IN]->(expanded:Chunk)
        RETURN DISTINCT expanded.file_path AS file_path, expanded.first AS first, expanded.last AS last
        """,
        {"chunks": seeds},
        database=database,
    )
    #exclude seed chunks from the result
    seed_set = {(s["file_path"], s["first"], s["last"]) for s in seeds}
    return [
        dict(r)
        for r in records
        if (r["file_path"], r["first"], r["last"]) not in seed_set
    ]
