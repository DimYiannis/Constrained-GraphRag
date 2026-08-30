"""
    extraction prompt for code chunks (Chunk.source_type == "code").

    Outlines' FSM guarantees the *shape* of the output matches
    ExtractionResult - it says nothing about whether the model picks the
    semantically right node/relation type for what it actually reads. This
    prompt's job is steering that semantic choice; schema.py enforces it
    can never be an invalid one.
"""

CODE_PROMPT_TEMPLATE = """\
You are extracting a knowledge graph from a chunk of source code.

Identify:
- entities: functions, classes, modules, or other named things defined or \
used in this chunk. Give each a name and a node_type - one of: \
Function, Class, Module, Concept, Entity.
- relationships: how those entities relate to each other, as \
(subject, relation, target) triples. Valid relations:
  - CALLS: one function invokes another
  - IMPORTS: a module imports another module
  - INHERITS_FROM: a class inherits from another class
  - DEFINED_IN: a function or class is defined in a module
  - RELATES_TO: a general association not covered by the above
  - REFERENCES: this chunk names a specific function/class defined \
elsewhere, without calling it directly

Only extract entities and relationships actually present in this chunk's \
text. Do not invent things that aren't there, and do not extract the file \
or chunk itself as an entity.

File: {file_path}

Code:
{text}
"""


def build_prompt(file_path: str, text: str) -> str:
    return CODE_PROMPT_TEMPLATE.format(file_path=file_path, text=text)
