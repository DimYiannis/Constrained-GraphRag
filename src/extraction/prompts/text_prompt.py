"""
    extraction prompt for text chunks (Chunk.source_type == "text").

    Covers markdown/docs prose AND config/data files (.yaml, .json, .sh, ...)
    that default to "text" - worded generically enough to degrade
    reasonably for both, not just narrative prose.
"""

TEXT_PROMPT_TEMPLATE = """\
You are extracting a knowledge graph from a chunk of text. It may be \
documentation prose, or a structured file such as configuration or data.

Identify:
- entities: named concepts, components, or things discussed in this \
chunk. Give each a name (the actual name as it appears in the text, e.g. \
"LoRAConfig", "enable_lora" - never the literal words "Function", "Class", \
"Module", "Concept", or "Entity", which are only the allowed values for \
node_type, not names) and a node_type - one of: \
Function, Class, Module, Concept, Entity.
- relationships: how those entities relate to each other, as \
(subject, relation, target) triples. Valid relations:
  - REFERENCES: this chunk names a specific function/class/module defined \
elsewhere in the codebase
  - RELATES_TO: a general association between two concepts
  - DEFINED_IN: a component is defined or configured in a module or file
  - CALLS, IMPORTS, INHERITS_FROM: only use these if the text explicitly \
describes that code-level relationship (e.g. "X calls Y internally") - \
most text chunks won't have any of these

Example. Given this text:

    LoRA adapters are configured via LoRAConfig, which enable_lora() uses \
to set up adapter weights.

Correct extraction:
  entities: [{{"name": "LoRAConfig", "node_type": "Class"}}, \
{{"name": "enable_lora", "node_type": "Function"}}]
  relationships: [{{"subject": "enable_lora", "relation": "RELATES_TO", "target": "LoRAConfig"}}]

Only extract entities and relationships actually present in this chunk's \
text. Do not invent things that aren't there, and do not extract the file \
or chunk itself as an entity.

File: {file_path}

Text:
{text}
"""


def build_prompt(file_path: str, text: str) -> str:
    return TEXT_PROMPT_TEMPLATE.format(file_path=file_path, text=text)
