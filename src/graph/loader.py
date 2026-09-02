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
    # Chunk node —  run driver to add Chunk node 
    # using the chunk's own file_path/first/last
    run_query(
        driver,
        "MERGE (chunk:Chunk {file_path: $file_path, first: $first, last: $last}) "
        "SET chunk.source_type = $source_type",
        {
            "file_path": chunk.file_path,
            "first": chunk.first,
            "last": chunk.last,
            "source_type": chunk.source_type,
        },
        database=database,
    )

    entity_types = {entity.name: entity.node_type.value for entity in result.entities}
    
    # add each entity node in the graph and the MENTIONED_IN 
    # edge to the chunk, in the same query.
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
    # node-type lookup to make sure both entity nodes exist
    # connect those two nodes with the right relation type (schema)
    for relationship in result.relationships:
        subject_type = entity_types.get(relationship.subject, FALLBACK_NODE_TYPE)
        target_type = entity_types.get(relationship.target, FALLBACK_NODE_TYPE)
        run_query(
            driver,
            f"MERGE (subject:{subject_type} {{name: $subject}}) "
            f"MERGE (target:{target_type} {{name: $target}}) "
            f"MERGE (subject)-[:{relationship.relation.value}]->(target)",
            {"subject": relationship.subject, "target": relationship.target},
            database=database,
        )

