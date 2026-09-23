import re
from collections import defaultdict

from src.chunking.spans import Chunk
from src.extraction.schema import ExtractionResult
from src.graph.neo4j_client import run_query

# every extracted node carries this base label, identity is
# (:Entity {name_normalized}), the specific type is an extra label
ENTITY_LABEL = "Entity"


def ensure_graph_constraints(driver, database: str | None = None) -> None:
    """
    one node per normalized name (also the index MERGE uses), and an
    index for the Chunk span MERGE/MATCH. idempotent
    """
    run_query(
        driver,
        "CREATE CONSTRAINT entity_name_normalized IF NOT EXISTS "
        "FOR (e:Entity) REQUIRE e.name_normalized IS UNIQUE",
        database=database,
    )
    run_query(
        driver,
        "CREATE INDEX chunk_span IF NOT EXISTS FOR (c:Chunk) ON (c.file_path, c.first, c.last)",
        database=database,
    )


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

    identity = normalized name only: every node is
    MERGE (e:Entity {name_normalized}), so the same name from any
    chunk lands on one node, however the model typed it

    the extracted type (Function/Class/Module/Concept) is an extra
    label, disagreeing chunks can give a node several. relationship
    endpoints not listed as entities start with just :Entity

    Cypher can't parameterize labels/relation types, so rows are
    grouped by label / relation type: one UNWIND query per group
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
    chunk_key = {"file_path": chunk.file_path, "first": chunk.first, "last": chunk.last}

    # each entity node, its type label, and its MENTIONED_IN edge to the chunk
    entities_by_label: dict[str, list[dict]] = defaultdict(list)
    for entity in result.entities:
        entities_by_label[entity.node_type.value].append(
            {"name": entity.name, "name_normalized": _normalize(entity.name)}
        )
    for label, rows in entities_by_label.items():
        set_label = "" if label == ENTITY_LABEL else f"SET entity:{label} "
        run_query(
            driver,
            "MATCH (chunk:Chunk {file_path: $file_path, first: $first, last: $last}) "
            "UNWIND $rows AS row "
            f"MERGE (entity:{ENTITY_LABEL} {{name_normalized: row.name_normalized}}) "
            "ON CREATE SET entity.name = row.name "
            f"{set_label}"
            "MERGE (entity)-[:MENTIONED_IN]->(chunk)",
            {**chunk_key, "rows": rows},
            database=database,
        )

    # relationship edges between entities, by normalized name
    relationships_by_type: dict[str, list[dict]] = defaultdict(list)
    for relationship in result.relationships:
        relationships_by_type[relationship.relation.value].append(
            {
                "subject": relationship.subject,
                "subject_normalized": _normalize(relationship.subject),
                "target": relationship.target,
                "target_normalized": _normalize(relationship.target),
            }
        )
    for relation, rows in relationships_by_type.items():
        run_query(
            driver,
            "UNWIND $rows AS row "
            f"MERGE (subject:{ENTITY_LABEL} {{name_normalized: row.subject_normalized}}) "
            "ON CREATE SET subject.name = row.subject "
            f"MERGE (target:{ENTITY_LABEL} {{name_normalized: row.target_normalized}}) "
            "ON CREATE SET target.name = row.target "
            f"MERGE (subject)-[:{relation}]->(target)",
            {"rows": rows},
            database=database,
        )

    # example stored in db
    #  (:Entity:Class {name: "TokenizerGroup", name_normalized: "tokenizergroup"})
    #        -[:CALLS]->
    #  (:Entity:Function {name: "encode", name_normalized: "encode"})
