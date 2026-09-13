from src.extraction.schema import ExtractionResult
from src.chunking.spans import Chunk
from src.graph.neo4j_client import run_query
import re

FALLBACK_NODE_TYPE = "Entity"

def _normalize(name: str) -> str:
    """
        address differences (case, punctuation, whitespace)
        that a semantic term might have so that we manage to 
        include all the variants into one node

        example: TokenizerGroup / tokenizer_group / Tokenizer Group
        all three terms fall into the same node

        problem thats not solved:
            paraphrases: Acme Corp vs Acme Corporation
            abbreviations: RAG - Retrieval-Augmented Generation
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())

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
    # run driver to add Chunk node 
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

    entity_types = {
        _normalize(entity.name): entity.node_type.value for entity in result.entities
    }
    
    # add each entity node in the graph and the MENTIONED_IN 
    # edge to the chunk, in the same query.
    for entity in result.entities:
        run_query(
            driver,
            f"MERGE (entity:{entity.node_type.value} {{name_normalized: $name_normalized}}) "
            "ON CREATE SET entity.name = $name "
            "WITH entity "
            "MATCH (chunk:Chunk {file_path: $file_path, first: $first, last: $last}) "
            "MERGE (entity)-[:MENTIONED_IN]->(chunk)",
            {
                "name": entity.name,
                "name_normalized": _normalize(entity.name),
                "file_path": chunk.file_path,
                "first": chunk.first,
                "last":chunk.last,
            },
            database=database
        )
    # node-type lookup to make sure both entity nodes exist
    # connect those two nodes with the right relation type (schema)
    for relationship in result.relationships:
        subject_type = entity_types.get(_normalize(relationship.subject), FALLBACK_NODE_TYPE)
        target_type = entity_types.get(_normalize(relationship.target), FALLBACK_NODE_TYPE)
        run_query(
            driver,
            f"MERGE (subject:{subject_type} {{name_normalized: $subject_normalized}}) "
            "ON CREATE SET subject.name = $subject "
            f"MERGE (target:{target_type} {{name_normalized: $target_normalized}}) "
            "ON CREATE SET target.name = $target "
            f"MERGE (subject)-[:{relationship.relation.value}]->(target)",
            {
                "subject": relationship.subject,
                "subject_normalized": _normalize(relationship.subject),
                "target": relationship.target,
                "target_normalized": _normalize(relationship.target),
            },
            database=database,
        )

        # example stored in db
        #  (:Class {name: "TokenizerGroup", name_normalized: "tokenizergroup"})
        #        -[:CALLS]->
        #  (:Function {name: "encode", name_normalized: "encode"})

