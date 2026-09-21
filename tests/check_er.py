# PYTHONPATH=/Users/yiannis/Developer/graphrag uv run python tests/check_er.py
"""
    verifies entity-resolution normalization merges cosmetic name variants
    into one graph node, no touching any real data already in Neo4j.

    scoped, not destructive: test chunks/entities all carry a distinct
    marker prefix that can never collide with real corpus data, and
    cleanup only deletes nodes matching that prefix.
"""
import src.__main__
from dotenv import load_dotenv
load_dotenv()

from src.graph import neo4j_client
from src.chunking.spans import Chunk
from src.extraction.schema import ExtractionResult, ExtractedEntity, ExtractedRelationship, NodeType, RelationType
from src.graph import loader

TEST_MARKER = "zzztestmarker"


def _cleanup(driver):
    """delete only this script's own test data, never anything else"""
    neo4j_client.run_query(
        driver,
        "MATCH (n) WHERE n.name_normalized STARTS WITH $marker "
        "OR (n:Chunk AND n.file_path STARTS WITH $marker) "
        "DETACH DELETE n",
        {"marker": TEST_MARKER},
    )


driver = neo4j_client.get_driver()
_cleanup(driver)  # in case a prior run of this script left test data behind

chunk1 = Chunk(f"{TEST_MARKER}_a.py", 0, 10, "text1", "code")
result1 = ExtractionResult(
    entities=[ExtractedEntity(name=f"{TEST_MARKER}TokenizerGroup", node_type=NodeType.CLASS)],
    relationships=[ExtractedRelationship(
        subject=f"{TEST_MARKER}TokenizerGroup",
        relation=RelationType.CALLS,
        target=f"{TEST_MARKER}encode",
    )],
)

chunk2 = Chunk(f"{TEST_MARKER}_b.py", 0, 10, "text2", "code")
result2 = ExtractionResult(
    entities=[ExtractedEntity(name=f"{TEST_MARKER}tokenizer_group", node_type=NodeType.CLASS)],
    relationships=[ExtractedRelationship(
        subject=f"{TEST_MARKER}tokenizer_group",
        relation=RelationType.CALLS,
        target=f"{TEST_MARKER}encode",
    )],
)

loader.load_chunk(driver, chunk1, result1)
loader.load_chunk(driver, chunk2, result2)

records, _, _ = neo4j_client.run_query(
    driver,
    "MATCH (n) WHERE n.name_normalized STARTS WITH $marker "
    "RETURN labels(n) AS labels, n.name AS name, n.name_normalized AS norm",
    {"marker": TEST_MARKER},
)
for r in records:
    print(dict(r))

records, _, _ = neo4j_client.run_query(
    driver,
    "MATCH (n:Class) WHERE n.name_normalized STARTS WITH $marker RETURN count(n) AS c",
    {"marker": TEST_MARKER},
)
print("Class node count (expect 1, not 2):", records[0]["c"])

records, _, _ = neo4j_client.run_query(
    driver,
    "MATCH (a)-[r:CALLS]->(b) WHERE a.name_normalized STARTS WITH $marker RETURN count(r) AS c",
    {"marker": TEST_MARKER},
)
print("total CALLS edges from test data (expect 1, not 2):", records[0]["c"])

_cleanup(driver)  # leave the graph exactly as we found it
driver.close()
