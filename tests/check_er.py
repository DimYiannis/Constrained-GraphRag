# PYTHONPATH=/Users/yiannis/Developer/graphrag uv run python tests/check_er.py
import src.__main__
from dotenv import load_dotenv
load_dotenv()

from src.graph import neo4j_client
from src.chunking.spans import Chunk
from src.extraction.schema import ExtractionResult, ExtractedEntity, ExtractedRelationship, NodeType, RelationType
from src.graph import loader

driver = neo4j_client.get_driver()
neo4j_client.run_query(driver, 'MATCH (n) DETACH DELETE n')

chunk1 = Chunk('a.py', 0, 10, 'text1', 'code')
result1 = ExtractionResult(
    entities=[ExtractedEntity(name='TokenizerGroup', node_type=NodeType.CLASS)],
    relationships=[ExtractedRelationship(subject='TokenizerGroup', relation=RelationType.CALLS, target='encode')],
)

chunk2 = Chunk('b.py', 0, 10, 'text2', 'code')
result2 = ExtractionResult(
    entities=[ExtractedEntity(name='tokenizer_group', node_type=NodeType.CLASS)],
    relationships=[ExtractedRelationship(subject='tokenizer_group', relation=RelationType.CALLS, target='encode')],
)

loader.load_chunk(driver, chunk1, result1)
loader.load_chunk(driver, chunk2, result2)

records, _, _ = neo4j_client.run_query(driver, 'MATCH (n) RETURN labels(n) AS labels, n.name AS name, n.name_normalized AS norm')
for r in records:
    print(dict(r))

records, _, _ = neo4j_client.run_query(driver, 'MATCH (n:Class) RETURN count(n) AS c')
print('Class node count:', records[0]['c'])

records, _, _ = neo4j_client.run_query(driver, 'MATCH ()-[r:CALLS]->() RETURN count(r) AS c')
print('total CALLS edges:', records[0]['c'])

driver.close()