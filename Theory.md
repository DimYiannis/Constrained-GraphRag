# Project Notes — Constrained GraphRAG

Living document: concepts and architecture. 

---

## 1. The core idea, in one paragraph

Plain BM25 only finds chunks that share a *word* with the query — it has no
notion of relationship. A knowledge graph fixes that by recording relationships
between entities explicitly, so retrieval can hop from a chunk that matched the
query to a *related* chunk that shares zero words with it. Building that graph
requires an LLM to read every chunk and extract entities/relationships — but a
naive LLM extraction step either needs an expensive frontier model to be
reliable, or produces malformed/inconsistent output. Grammar-constrained
decoding closes that gap: forcing the schema at every generated token, not
just requesting it via prompt, makes a small model's structured output as
reliable as a frontier model's.

**In this project:** that bet is Qwen3-0.6B + Outlines-based constrained
decoding, extracting into a fixed graph schema (`schema.py`).

---

## 2. Concepts

### RAG (Retrieval-Augmented Generation)
Instead of relying on a model's parametric memory, fetch relevant text from an
external corpus at query time and put it in the prompt as grounding. Cuts
hallucination on knowledge the model wasn't reliably trained on.

### BM25
Ranks a chunk against a query by summing `idf(t) · tf-saturation(t, chunk)`
over every shared term: rare terms count for more (`idf`), and a term
repeating in a chunk gives diminishing returns rather than a linear reward
(`tf-saturation`, tuned by `k1`) — `b` controls how much a chunk's length gets
normalized against, so a long chunk can't win purely by containing more words.
Pure term overlap — no notion of meaning or relationship.

### Chunk
The unit of retrieval: a bounded span of a source document, never a whole
file, tracked by a location plus an offset range rather than by storing the
text itself — so the underlying source stays the single source of truth and
the index can always re-slice it instead of duplicating it.

**In this project:** `(file_path, first, last, text)`. Offsets are ground
truth — `text[first:last] == chunk.text` always holds, so chunk text is never
persisted in the index, only the offsets.

### Identifier-aware tokenizer
A generic subword/BPE tokenizer splits on statistical frequency, which is
tuned for natural language, not code — it can obscure the exact identifier a
query is quoting. An identifier-aware tokenizer instead splits on
non-alphanumerics and emits an identifier both whole *and* as its
CamelCase/snake_case subtokens, so a query can quote the whole identifier or
paraphrase part of it and still match.

**In this project:** `enable_lora` → `enable_lora`, `enable`, `lora`.

### Why graph expansion, concretely
Lexical retrieval only finds chunks that share a *surface word* with the
query. Content that implements the same idea under different vocabulary is
invisible to it — no shared words, no match. A knowledge graph fixes this by
storing the connection explicitly as an edge, so retrieval can hop to related
content it has zero lexical overlap with.

**In this project:** BM25 finds chunks mentioning "LoRA" and "loading". A
chunk implementing the actual mechanism under a totally different name
(`apply_adapter_weights`, no mention of "LoRA" anywhere) is invisible to BM25.
If the graph knows `enable_lora CALLS apply_adapter_weights`, retrieval can
hop that edge and pull the second chunk in anyway.

### Schema-constrained extraction and taxonomies
A *taxonomy*, here, means a small, fixed vocabulary of node/relationship
types decided before any extraction runs, rather than discovered from the
corpus. Fixing that vocabulary up front and enforcing it mechanically (not
just requesting it via prompt) keeps type-labeling *consistent* across
independent extraction calls — without it, one chunk's extraction might call
something a `"Function"` and another chunk's extraction might call the same
kind of thing a `"Method"`, and that drift compounds across a whole corpus.

A taxonomy is the vocabulary, though, not the linking mechanism — it doesn't
by itself connect anything. Two different mentions of the same real-world
entity still need to be recognized as *the same node*, which is a separate
problem (**entity resolution**): matching by exact name is cheap and handles
the common case, but does nothing for `"Acme Corp"` vs `"Acme Corporation"` —
that needs actual normalization or fuzzy matching. A taxonomy only guarantees
two extractions are *allowed* to agree on an entity's type, so a merge is
possible in the first place; it says nothing about whether their names will
actually match.

**In this project:** node types (`Function`, `Class`, `Module`, `Concept`,
`Entity`, `Chunk`) and relationship types (`CALLS`, `IMPORTS`,
`INHERITS_FROM`, `DEFINED_IN`, `MENTIONED_IN`, `RELATES_TO`, `REFERENCES`) are
fixed in `schema.py` and enforced at generation time via enum-constrained
fields. Cross-chunk linking happens later, in `loader.py`: two chunks that
both extract an entity named `"TokenizerGroup"` (exact string match) get
merged into the same graph node via `MERGE`. Fuzzy/normalized entity
resolution beyond exact-match isn't built yet — a genuinely open problem
(tracked in `progress.md`).

**Closed taxonomy vs. open labelling, more generally.** The taxonomy approach
above is one end of a spectrum; the other end is open extraction, the style
used by, e.g., Microsoft's original GraphRAG, where the model free-labels
relationships in its own words (`relationship_description`) instead of
picking from a closed set. Open extraction is more expressive — it can
describe a relationship a closed taxonomy has no label for — but it pays for
that with duplication (`"depends on"` / `"relies on"` / `"requires"` all
meaning the same edge) that needs a later clustering/dedup pass to clean up,
and it has no way to be validated at generation time: nothing stops the model
from emitting free text that doesn't correspond to anything at all. A closed
taxonomy trades away that expressiveness for a small, enumerable universe
that grammar-constrained decoding can mechanically enforce, token by token —
which matters specifically when reliability can't come from the model just
being smart enough. Open extraction and a small model don't mix well for
that reason: there's nothing fixed to constrain generation against.

### Grammar-constrained decoding — the actual mechanism
Instead of asking a model to produce valid structured output and hoping, the
target schema is compiled into a grammar. At every generated token, the
decoder computes which next tokens would keep the output on a path toward
valid schema-conforming output, and masks every other token's probability to
zero before sampling. The model is not being polite about following
instructions — a schema-violating token is never a candidate to begin with.
This is why a small model can be trustworthy at structured extraction: the
reliability gap that would normally require a frontier model is closed by
making invalid output structurally impossible, not by making the model
smarter.

### Bridging the graph back to source text
An entity-relationship graph on its own is just a web of concepts with no way
back to the text it came from. Fixing that requires the source span itself to
be a node in the graph, with every extracted entity linked back to the
span(s) it was mentioned in — that link is what turns "a graph of concepts"
into "a graph that can hand retrieval real, quotable text."

**In this project:** `Chunk` is always its own graph node; every extracted
entity links to it via `MENTIONED_IN`.

### Graph databases
A graph database is general-purpose storage plus a query language built
around traversing relationships directly, rather than joining rows across
tables — the same conceptual role a relational database plays for tabular
data. It typically has no separate built-in "traversal" feature; a multi-hop
expansion is just a query, structurally no different from any other read.

**In this project:** Neo4j + Cypher. A 1-2 hop expansion over
`CALLS`/`RELATES_TO`/`REFERENCES` is a single Cypher query, sent the same way
any other read would be.

### Query-time flow (retrieve → expand → answer)
The general pattern for retrieval + graph expansion: get a seed set from
whatever retrieval method is in use, map that seed set into the graph,
traverse outward a bounded number of hops, map the newly-reached graph nodes
back to retrievable text, then merge everything into the final context.

**In this project, concretely:**
1. BM25 returns top-k chunks for the query.
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

### Caching
Memoizing expensive, repeatable work — a model call, an index lookup — keyed
by its input, so a repeat (or near-duplicate) request skips redoing it.
Worth it exactly when the same expensive computation recurs often enough
that the storage cost pays for itself in time saved; not worth it for
work that's already cheap or never repeats.

**In this project:** planned for repeat/near-duplicate extraction calls
and/or repeat queries. `cache/cache.py` is currently a placeholder — not
yet designed.

### Batching
Running inputs through a model one at a time usually leaves the underlying
hardware (GPU/CPU vector units) underused — most of the fixed cost of a
forward pass gets paid whether it processes one sequence or several at once.
Batching groups multiple inputs into a single call so that cost is amortized
across all of them, at the cost of some complexity (padding variable-length
inputs to a common length so they can share one call).

**In this project:** extraction currently runs one chunk per `generate()`
call, fully sequential.

### Semantic (dense) retrieval
Instead of matching on shared surface words, embed both the query and each
chunk into a shared vector space and rank by similarity. This captures
*meaning* rather than literal term overlap — it can find a chunk that means
the same thing as the query even if it shares none of the same words. The
tradeoff runs the other way from BM25: dense retrieval is weaker on exact
identifiers and rare terms, since embeddings blur precise tokens together
in a way pure term-matching doesn't.


### Hybrid fusion (RRF)
Combining rankings from two different retrieval methods (e.g. lexical +
dense) into one list is awkward if you try to do it by combining their raw
scores directly — a BM25 score and a cosine similarity aren't on comparable
scales. Reciprocal Rank Fusion sidesteps this by scoring each document using
only its *rank* in each list (`1/(k + rank)`, summed across lists) — a
document that ranks well in either method gets boosted, without either
method dominating just because its numbers happen to be bigger.

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
