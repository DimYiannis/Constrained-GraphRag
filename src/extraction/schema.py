"""
    graph schema: node/relationship types, and the Pydantic model handed
    directly to Outlines for grammar-constrained extraction. This IS the
    grammar source, not just documentation of it - Outlines compiles
    ExtractionResult's JSON schema into the FSM that masks the model's
    logits at every generated token.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class NodeType(str, Enum):
    FUNCTION = "Function"
    CLASS = "Class"
    MODULE = "Module"
    CONCEPT = "Concept"
    ENTITY = "Entity"
    CHUNK = "Chunk"


class RelationType(str, Enum):
    CALLS = "CALLS"
    IMPORTS = "IMPORTS"
    INHERITS_FROM = "INHERITS_FROM"
    DEFINED_IN = "DEFINED_IN"
    MENTIONED_IN = "MENTIONED_IN"
    RELATES_TO = "RELATES_TO"
    REFERENCES = "REFERENCES"


# CHUNK is never something the model extracts - Chunk nodes already exist
# from the chunking phase, before extraction runs.
ExtractableNodeType = Literal[
    NodeType.FUNCTION,
    NodeType.CLASS,
    NodeType.MODULE,
    NodeType.CONCEPT,
    NodeType.ENTITY,
]

# MENTIONED_IN is never emitted by the model either - loader.py adds it
# automatically for every entity extracted from a given chunk, since it's
# structurally implied (an entity extracted FROM a chunk is trivially
# mentioned in it) rather than something worth spending the model's
# constrained generation budget on.
ExtractableRelationType = Literal[
    RelationType.CALLS,
    RelationType.IMPORTS,
    RelationType.INHERITS_FROM,
    RelationType.DEFINED_IN,
    RelationType.RELATES_TO,
    RelationType.REFERENCES,
]


class ExtractedEntity(BaseModel):
    name: str
    node_type: ExtractableNodeType


class ExtractedRelationship(BaseModel):
    subject: str
    relation: ExtractableRelationType
    target: str


class ExtractionResult(BaseModel):
    """
        one chunk's extraction output - the object passed to
        outlines.generate.json(model, ExtractionResult).
    """

    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]
