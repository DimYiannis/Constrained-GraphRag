# Project Notes — Constrained GraphRAG

Living document: concepts and architecture. Add to the concept sections
whenever something clicks or gets decided differently than planned. This file
is for understanding and owning the project, not for the portfolio —
README.md is the polished version. Build status and the decisions log live in
`progress.md`.

---

## 1. The core idea, in one paragraph

Plain BM25 only finds chunks that share a *word* with the query — it has no
notion of relationship. A knowledge graph fixes that by recording relationships
between entities explicitly, so retrieval can hop from a chunk that matched the
query to a *related* chunk that shares zero words with it. Building that graph
requires an LLM to read every chunk and extract entities/relationships — but a
naive LLM extraction step either needs an expensive frontier model to be
reliable, or produces malformed/inconsistent output. This project's bet:
**grammar-constrained decoding lets a small model (Qwen3-0.6B) be as reliable
as a frontier model at structured extraction**, because the schema is enforced
mechanically at every generated token, not just requested via prompt.

---

## 2. Concepts

### RAG (Retrieval-Augmented Generation)
Instead of relying on a model's parametric memory, fetch relevant text from an
external corpus at query time and put it in the prompt as grounding. Cuts
hallucination on knowledge the model wasn't reliably trained on.

### BM25
Bag-of-words ranking: scores a chunk for a query as
`Σ idf(t) · tf-saturation(t, chunk)` over shared terms. `k1` controls how fast
term-frequency saturates (diminishing returns for repeating a term); `b`
controls how strongly chunk length is normalized against. No notion of meaning
or relationship — pure term overlap.

### Chunk
The unit of retrieval: `(file_path, first, last, text)`. Never a whole file.
Offsets are ground truth — `text[first:last] == chunk.text` always holds, so
chunk text is never persisted in the index, only the offsets.

### Identifier-aware tokenizer
Not a subword/BPE tokenizer. Splits on non-alphanumerics, lowercases, and
emits identifiers both whole *and* as CamelCase/snake_case subtokens
(`enable_lora` → `enable_lora`, `enable`, `lora`) — so a query can quote a
whole identifier or paraphrase it and still match.

### Why graph expansion, concretely
BM25 finds chunks mentioning "LoRA" and "loading". A chunk implementing the
actual mechanism under a totally different name (`apply_adapter_weights`, no
mention of "LoRA" anywhere) is invisible to BM25 — no shared words. If the
graph knows `enable_lora CALLS apply_adapter_weights`, retrieval can hop that
edge and pull the second chunk in anyway.

### Schema-constrained extraction
Before extraction starts, `schema.py` fixes the entire universe: which node
types exist (`Function`, `Class`, `Module`, `Concept`, `Entity`, `Chunk`) and
which relationship types exist (`CALLS`, `IMPORTS`, `INHERITS_FROM`,
`DEFINED_IN`, `MENTIONED_IN`, `RELATES_TO`, `REFERENCES`). Nothing outside
that list is a legal extraction result — this is enforced mechanically, at
generation time, because `node_type`/`relation` are enum-constrained fields.

**Correction (this doc previously overstated this):** constrained decoding
does *not* solve the `"Acme Corp"` vs `"Acme Corporation"` duplicate-entity
problem. That problem is about the free-text `name` field on
`ExtractedEntity`, which has no enum constraint — it can't have one, since
entity names come from actual corpus content and aren't a small fixed
vocabulary the way node/relation types are. Nothing stops two independent
extraction calls from naming the same real thing two different ways.

**`NodeType`/`RelationType` are the vocabulary, not the linking mechanism.**
They don't directly "connect chunks" — they're the closed vocabulary that
makes extraction *consistent* across chunks in one specific dimension: type.
Without a fixed set, one chunk's extraction might call something a
`"Function"` and another chunk's extraction might call the same kind of
thing a `"Method"` — that drift, and only that drift, is what fixing the
vocabulary up front prevents.

**The actual cross-chunk connection happens later, in the loader — not in
`schema.py`.** When two different chunks both extract an entity named
`"TokenizerGroup"` (same exact string), `loader.py` (not yet built) matches
them by name and merges them into the *same* graph node — that shared node
is what actually links the two chunks together, via a `MENTIONED_IN` edge
from each. But "matches them by name" as just described only handles
*exact* string matches. It does nothing for `"Acme Corp"` vs
`"Acme Corporation"` — that needs actual entity resolution (normalization,
fuzzy matching, or similar) inside `loader.py`, which isn't designed yet.
`schema.py`'s job is narrower than it might look: it only guarantees two
extractions are *allowed* to agree on an entity's type, so a merge is
possible in the first place — it says nothing about whether their names will
actually match.

### Grammar-constrained decoding — the actual mechanism
Instead of asking the model to produce valid JSON and hoping, the schema is
compiled into a grammar. At every generated token, the decoder computes which
next tokens would keep the output on a path toward valid schema-conforming
output, and masks every other token's probability to zero before sampling.
The model is not being polite about following instructions — a
schema-violating token is never a candidate to begin with. This is *why* a
0.6B model is trustworthy here: the reliability gap that would normally
require a frontier model is closed by making invalid output structurally
impossible, not by making the model smarter.

### The `Chunk` node and `MENTIONED_IN`
`Chunk` is always its own graph node. Every extracted entity links back to the
chunk(s) it was mentioned in via `MENTIONED_IN`. This edge is the bridge
between "graph of concepts" and "actual retrievable text" — without it, the
graph would be a web of entities with no way to get back to source spans.

### Query-time flow (retrieve → expand → answer)
1. BM25 returns top-k chunks for the query (**built, working today**).
2. Walk `MENTIONED_IN` *backward* from those chunks → seed entities mentioned
   in them.
3. Traverse 1-2 hops outward from seeds over `CALLS`/`RELATES_TO`/`REFERENCES`
   → reach other entities.
4. Walk `MENTIONED_IN` *forward* from those new entities → new chunks. This is
   the set BM25 alone would have missed.
5. Dedupe, merge original + graph-expanded chunks, build the final prompt.

### Concrete example (from this project's actual corpus)
Chunk: `tests/lora/test_tokenizer_group.py` (a real top hit for
`search "enable lora"` today). Extraction over it would produce triples like:

```
(TokenizerGroup, DEFINED_IN, tokenizer_group.py)
(TokenizerGroup, CALLS, encode)
(test_tokenizer_group, REFERENCES, TokenizerGroup)
(TokenizerGroup, MENTIONED_IN, <this chunk's id>)
```

Loaded into Neo4j: `TokenizerGroup` becomes a `Class` node, `encode` a
`Function` node, edges between them, and `TokenizerGroup` gets a
`MENTIONED_IN` edge to the chunk it came from.

---

## 3. Architecture — where each piece fits

| Module | Job | Depends on |
|---|---|---|
| `extraction/schema.py` | Fixed node/relationship vocabulary; grammar source for constrained decoding *and* what the Neo4j loader writes against | nothing — build first |
| `extraction/extractor.py` | Runs Qwen3-0.6B + constrained decoder over each chunk → triples. Needs separate prompts for code vs text chunks | `schema.py`, `extraction/prompts/`, `Chunk.source_type` |
| `extraction/prompts/` | Code-chunk prompt vs text-chunk prompt | `schema.py` |
| `graph/neo4j_client.py` | Connection/session handling | nothing |
| `graph/loader.py` | Writes extracted triples + `Chunk` nodes + `MENTIONED_IN` edges into Neo4j | `neo4j_client.py`, `schema.py`, `extractor.py`'s output shape |
| `graph/traversal.py` | Cypher for query-time steps 2-4 above | `neo4j_client.py`, loaded graph |
| `pipeline/index_pipeline.py` | Orchestrates offline: corpus → chunk → extract → load graph | all of the above + `chunking/`, `retrieval/lexical/` |
| `pipeline/query_pipeline.py` | Orchestrates runtime: query → retrieve → graph expand → prompt → answer | `retrieval/lexical/`, `graph/traversal.py` |

---

## 4. Open questions / things to decide when we get there

- How does `extractor.py` get chunks — re-chunk from `data_dir` itself
  (mirrors how `lexical.build_index()` works today), or take a pre-chunked
  list produced once by `index_pipeline.py` and shared with `lexical.py`?
- What does "validated end-to-end" mean concretely for the lexical+graph
  milestone before dense retrieval is allowed to start? Some kind of
  evaluation script needs to exist first (`evaluation/` is empty).
- ~~Constrained decoding implementation: Outlines vs XGrammar vs something
  else~~ — **settled: Outlines.** Verified against the actual installed
  package (`1.3.3`), not docs (one docs page turned out stale —
  `outlines.generate` doesn't exist in this version). Confirmed API:
  `outlines.from_transformers(hf_model, hf_tokenizer)` to wrap a HF model,
  `outlines.Generator(model, output_type=ExtractionResult)` built once and
  reused per-chunk (avoids re-compiling the FSM on every one of ~28k calls).
  `transformers`/`torch` still need adding as deps before this runs for real.
- **Entity resolution / name normalization in `loader.py`** — genuinely
  unsolved, not just unbuilt. `loader.py`'s planned exact-name matching does
  nothing for `"Acme Corp"` vs `"Acme Corporation"` (see the correction in
  §2, "Schema-constrained extraction"). Options to weigh when we get there:
  cheap normalization (lowercase/strip punctuation) as a `MERGE` key, fuzzy
  string matching (edit distance) at load time, or asking the model itself
  to canonicalize names during extraction. Each has a different cost/recall
  tradeoff and none is decided.
