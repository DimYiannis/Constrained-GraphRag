from src.extraction.schema import ExtractionResult
from src.chunking.spans import Chunk
from src.graph.neo4j_client import run_query


FALLBACK_NODE_TYPE = "Entity"

def load_chunk(
    driver,
    chunk: Chunk,
    result: ExtractionResult,
    database: str | None = None,
) -> None:
    """
        write one chunk's extraction into neo4j

        loading each chunk at a time,
        - MERGE (entity:{type} {name: $name}) checks 
          against everything already in the graph every 
          single time it runs
    """
    run_query(
        driver,
        "MERGE (c:Chunk {file_path: $file_path, first: $first, last: $last}) "
        "SET c.source_type = $source_type",
        {
            "file_path": chunk.file_path,
            "first": chunk.first,
            "last": chunk.last,
            "source_type": chunk.source_type,
        },
        database=database,
    )

    entity_types = {e.name: e.node_type.value for e in result.entities}

    for entity in result.entities:
        run_query(
            driver,
            f"MERGE (entity:{entity.node_type.value} {{name: $name}}) "
            "WITH entity "
            "MATCH (chunk:Chunk {file_path: $file_path, first: $first, last: $last}) "
            "MERGE (entity)-[:MENTIONED_IN]->(chunk)",
            {
                "name": entity.name,
                "file_path": chunk.file_path,
                "first": chunk.first,
                "last":chunk.last,
            },
            database=database
        )
    
    for rel in result.relationships:
        subject_type = entity_types.get(rel.subject, FALLBACK_NODE_TYPE)
        target_type = entity_types.get(rel.target, FALLBACK_NODE_TYPE)
        run_query(
            driver,
            f"MERGE (subject:{subject_type} {{name: $subject}}) "
            f"MERGE (target:{target_type} {{name: $target}}) "
            f"MERGE (subject)-[:{rel.relation.value}]->(target)",
            {"subject": rel.subject, "target": rel.target},
            database=database,
        )

