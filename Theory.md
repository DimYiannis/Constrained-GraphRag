# Project Notes — Constrained GraphRAG

Living document: concepts and architecture. 

## Contents

- [1. The core idea, in one paragraph](#1-the-core-idea-in-one-paragraph)
- [2. Cheat sheet](#2-cheat-sheet)
  - [Retrieval](#retrieval)
  - [Extraction](#extraction)
  - [Graph](#graph)
  - [Evaluation](#evaluation)
  - [Bugs found and fixed](#bugs-found-and-fixed)
- [3. Concepts](#3-concepts)
  - [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
  - [BM25](#bm25)
  - [Chunk](#chunk)
  - [Identifier-aware tokenizer](#identifier-aware-tokenizer)
  - [Why graph expansion, concretely](#why-graph-expansion-concretely)
  - [Schema-constrained extraction and taxonomies](#schema-constrained-extraction-and-taxonomies)
  - [Grammar-constrained decoding — the actual mechanism](#grammar-constrained-decoding--the-actual-mechanism)
  - [Bridging the graph back to source text](#bridging-the-graph-back-to-source-text)
  - [Graph databases](#graph-databases)
  - [Query-time flow (retrieve → expand → answer)](#query-time-flow-retrieve--expand--answer)
  - [Concrete example (from this project's actual corpus)](#concrete-example-from-this-projects-actual-corpus)
  - [Caching](#caching)
  - [Batching](#batching)
  - [Semantic (dense) retrieval](#semantic-dense-retrieval)
  - [Hybrid fusion (RRF)](#hybrid-fusion-rrf)
  - [Deterministic top-k](#deterministic-top-k)
  - [Property graphs, identity, and `MERGE`](#property-graphs-identity-and-merge)
  - [Entity resolution in practice](#entity-resolution-in-practice)
  - [Graph traversal: paths, distance, ranking](#graph-traversal-paths-distance-ranking)
  - [Evaluating retrieval honestly](#evaluating-retrieval-honestly)
- [4. Architecture — where each piece fits](#4-architecture--where-each-piece-fits)
- [5. What the project showed](#5-what-the-project-showed)

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

## 2. Cheat sheet

The whole project on one screen. Each row is expanded in [Concepts](#3-concepts).

### Retrieval
| Concept | What it is | How it shows up in this project |
|---|---|---|
| **BM25** | Scores documents by term matches, weighted by **IDF** (rare terms count more), with **term-frequency saturation** (`k1`: the 10th repeat adds little) and **length normalization** (`b`: long chunks aren't favoured) | Strongest retriever here (0.660 recall@3), because answers depend on exact identifiers |
| **Identifier-aware tokenizer** | Keeps the whole identifier *and* its parts: `enable_lora` → `enable_lora`, `enable`, `lora`; `TokenizerGroup` → `tokenizergroup`, `tokenizer`, `group` | Why BM25 already matches spelling variants |
| **Dense retrieval** | Encode text as vectors, rank by cosine similarity; with L2-normalized vectors, cosine = dot product | `all-MiniLM-L6-v2`, 384 dimensions. Weaker on code (0.355@3): embeddings capture meaning, not exact names |
| **RRF** | `score = Σ 1/(c + rank)` over both rankings. Uses **ranks, not scores**, because BM25 scores and cosine similarities are on different scales | `c = 60`, 100 candidates per retriever. Hybrid never beat lexical overall |
| **Deterministic top-k** | Ties broken by the lower id across the whole score array, so `top_k(n)` is a prefix of `top_k(n+m)` | Required for `@matched` to be a fair comparison |

### Extraction
| Concept | What it is | How it shows up in this project |
|---|---|---|
| **Constrained decoding** | The JSON schema is compiled into a **finite state machine**; at every step, tokens that would break the grammar get probability 0 | A 0.6B model produces valid structure every time, with no reject-and-retry loop |
| **What it doesn't guarantee** | Correctness. Only *structure* is enforced, not *truth* | Fabricated names (`from_vllm.utils.lrucache`), wrong types (72% of entities `Entity`-only), junk names (`self`) |
| **The truncation gap** | Hitting `max_new_tokens` mid-JSON gives output that's valid so far but incomplete | 8 of 303 chunks failed; `model_validate_json` catches it and the chunk is skipped |
| **Regex constraints** | `^[A-Za-z_][A-Za-z0-9_.]*$` on names, compiled into the same FSM | Stopped whole import statements being emitted as names |
| **Closed taxonomy** | A fixed set of relation types instead of free-text labels | Avoids label proliferation ("calls", "invokes", "is called by"); `RELATES_TO` is the deliberate catch-all |

### Graph
| Concept | What it is | How it shows up in this project |
|---|---|---|
| **Property graph** | Nodes with labels and properties, typed edges | `Chunk`, `:Entity:Function`, …; `CALLS`, `MENTIONED_IN`, … |
| **`MERGE` semantics** | Matches on **label + properties**; creates the node if there's no match | Why keying on the type label split 78 names across 165 nodes |
| **Entity resolution** | Deciding two mentions are the same thing: **deterministic** (exact key) vs **probabilistic/fuzzy** | Normalized name as the identity key; deliberately no fuzzy merging, because a wrong merge is silent |
| **Constraints / indexes** | A uniqueness constraint enforces identity in the database; an index speeds up `MERGE`/`MATCH` | `ensure_graph_constraints()`. Outlines can't enforce uniqueness *across* chunks |
| **`UNWIND` batching** | One query per batch instead of one per row | Labels can't be query parameters, so rows are grouped per label/relation type, and only enum values are interpolated (injection-safe) |
| **Variable-length traversal** | `-[:A\|B*0..2]-`: paths of 0–2 edges, ignoring direction | distance = `min(length(path))`; ranked by distance, then support; seeds excluded before `LIMIT`; `max_expanded` caps hub fan-out |

### Evaluation
| Concept | Meaning |
|---|---|
| **recall@k** | Share of questions where a retrieved chunk covers ≥50% of the answer span |
| **Ablation study** | Remove or replace one component, hold the rest fixed, measure the effect |
| **Confound** | A second difference between arms that explains the result on its own. Here: +graph got more chunks |
| **Budget-matched baseline** (`@matched`) | Same retriever, same number of chunks. Isolates *which* chunks the graph picks from *how many* |
| **Random-expansion control** (`+random`) | Same number of chunks, picked at random from the graph. Rules out "every chunk in the graph is an answer" |
| **Leakage / upper bound** | A graph built only from answer chunks inflates results; distractors are the realistic test |
| **Reach vs usefulness** | The graph can reach chunks BM25 can't (reach); `@matched` tests whether they're *better* (usefulness) |

### Bugs found and fixed
| Bug | Symptom | Fix |
|---|---|---|
| Eval confound | Lexical +0.045 from the graph | It was k chunks vs k+50; added `@matched` and capped expansion at k |
| Identity split | 78 names across 165 nodes; dead-end fallback nodes | Identity = normalized name only, type as extra label; migrated in place with a backup |
| Traversal started at 1 hop | Chunks sharing the seed's entity never returned | `*1..hops` → `*0..hops` |
| Cap before seed exclusion | Fewer than `max_expanded` new chunks | Exclude seeds inside the query, before `LIMIT`; rank by distance/support |
| Non-deterministic top-k | `top_k(5)` not extending `top_k(3)`; lexical@3 0.665 | Global tie-break by chunk id (0.660, stable) |
| Garbage names | Whole import statements as entity names | Regex in the grammar, not just the prompt |
| Blanket delete | `MATCH (n) DETACH DELETE n` wiped the graph | Marker-prefixed test data, scoped cleanup only |

---

## 3. Concepts

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

**In this project:** `(file_path, first, last, text, source_type)`, with
`file_path` relative to the corpus root and `source_type` "code" or "text"
(picks the extraction prompt). Offsets are ground truth — the file's
`text[first:last] == chunk.text` always holds, so chunk text is never
persisted in the index or the graph, only the offsets. A chunk's identity
everywhere (BM25 rows, graph `Chunk` nodes, ground truth) is
`(file_path, first, last)`, which is why changing chunking forces a rebuild.

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
fields (the model may only emit the `Extractable*` subsets: `Chunk` and
`MENTIONED_IN` are structural, added by the loader). Cross-chunk linking
happens later, in `loader.py`: entities merge on a *normalized* name
(lowercase, strip everything but `[a-z0-9]`), so `"TokenizerGroup"` and
`"tokenizer_group"` from two chunks become one node. Paraphrases and
abbreviations (`Acme Corp` / `Acme Corporation`) are deliberately *not*
fuzzy-merged. See "Entity resolution in practice" below.

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

**What it does *not* guarantee:**
- **Correctness.** The grammar enforces *shape*, not *truth*. A name can be
  identifier-shaped and fabricated (`from_vllm.utils.lrucache`); a type can be
  valid and wrong (72% of entities here ended up as the catch-all `Entity`);
  a name can be valid and useless (`self`, `with`, `K`).
- **Completeness.** If generation hits `max_new_tokens` mid-JSON, the output
  is valid *so far* but truncated. The FSM can't prevent that, so validation
  after generation is still needed: `extract()` runs `model_validate_json`
  and skips chunks that fail (8 of 303 here).

**Two layers of constraint, and why the second one mattered:** enums fix
`node_type`/`relation`; a regex (`^[A-Za-z_][A-Za-z0-9_.]*$`) on
`name`/`subject`/`target` is compiled into the same FSM. Prompt wording alone
only partly stopped the model emitting whole import statements as names; the
regex made them unsampleable. Rule of thumb: if something must never happen,
put it in the grammar, not the prompt.

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

**In this project:** Neo4j + Cypher. The whole 0-2 hop expansion over the six
entity-to-entity relationship types is a single Cypher query, sent the same
way any other read would be.

### Query-time flow (retrieve → expand → answer)
The general pattern for retrieval + graph expansion: get a seed set from
whatever retrieval method is in use, map that seed set into the graph,
traverse outward a bounded number of hops, map the newly-reached graph nodes
back to retrievable text, then merge everything into the final context.

**In this project, concretely:**
1. A retriever returns top-k seed chunks (`answer` uses BM25; the evaluation
   also runs dense and hybrid).
2. Follow `MENTIONED_IN` into those chunks (edges point entity → chunk) to get
   the seed entities.
3. Traverse `0..hops` (default 2) entity-to-entity edges over `CALLS`,
   `IMPORTS`, `INHERITS_FROM`, `DEFINED_IN`, `RELATES_TO`, `REFERENCES`,
   ignoring direction. 0 hops = the seed entity itself.
4. Follow `MENTIONED_IN` out of the reached entities → new chunks. Seeds are
   excluded, results ranked (see "Graph traversal" below) and capped at
   `max_expanded` (50 in `answer`, k in the evaluation).
5. Merge seed + expanded chunks, re-slice their text from source by offset,
   build the prompt, generate.

### Concrete example (from this project's actual corpus)
Chunk: `tests/lora/test_tokenizer_group.py` (a real top hit for
`search "enable lora"` today). Extraction over it would produce triples like:

```
(TokenizerGroup, DEFINED_IN, tokenizer_group.py)
(TokenizerGroup, CALLS, encode)
(test_tokenizer_group, REFERENCES, TokenizerGroup)
(TokenizerGroup, MENTIONED_IN, <this chunk's id>)
```

Loaded into Neo4j: `TokenizerGroup` becomes an `(:Entity:Class)` node,
`encode` an `(:Entity:Function)` node, edges between them, and
`TokenizerGroup` gets a `MENTIONED_IN` edge pointing to the chunk it came
from. Every entity carries the base `:Entity` label; the extracted type is an
extra label.

### Caching
Memoizing expensive, repeatable work — a model call, an index lookup — keyed
by its input, so a repeat (or near-duplicate) request skips redoing it.
Worth it exactly when the same expensive computation recurs often enough
that the storage cost pays for itself in time saved; not worth it for
work that's already cheap or never repeats.

**In this project:** a disk-backed, exact-match cache keyed on
`sha256(query, k, hops, max_new_tokens)`. Two weaknesses worth knowing:
`"enable lora"` and `"Enable LoRA"` miss each other (a *semantic cache* would
embed queries and match by similarity), and the key omits things that change
the answer (`max_expanded`, the graph's contents), so a stale answer can be
served after those change. General lesson: a cache key must include every
input that affects the output.


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

**In this project:** `all-MiniLM-L6-v2` (384 dimensions). Vectors are
L2-normalized once, so cosine similarity reduces to a plain dot product
(`embeddings @ query_vector`). It scored 0.355 recall@3 vs BM25's 0.660: on a
codebase, questions hinge on exact identifiers, which is BM25's strength.


### Hybrid fusion (RRF)
Combining rankings from two different retrieval methods (e.g. lexical +
dense) into one list is awkward if you try to do it by combining their raw
scores directly — a BM25 score and a cosine similarity aren't on comparable
scales. Reciprocal Rank Fusion sidesteps this by scoring each document using
only its *rank* in each list (`1/(c + rank)`, summed across lists) — a
document that ranks well in either method gets boosted, without either
method dominating just because its numbers happen to be bigger.

**In this project:** `c = 60` (the original paper's constant), with 100
candidates pulled from each retriever before fusing down to k, so a chunk
ranked low by one retriever but high by the other can still surface. Hybrid
never beat lexical overall here: fusing in a much weaker retriever pulled
lexical's ranking down. Fusion helps when the two retrievers are comparably
good and make *different* mistakes.

### Deterministic top-k
Picking the k best scores sounds trivial until scores tie. A partial sort
(`argpartition`) returns *some* k of the tied items, not a stable choice, so
the same query could return different chunks across runs, and `top_k(5)`
might not start with the same 3 chunks as `top_k(3)`.

**In this project:** `retrieval/ranking.py` breaks ties by the lower chunk id
across the whole score array (partial select, pull in every chunk tied with
the k-th score, then `lexsort` by score desc, id asc). That guarantees
`top_k(n)` is a prefix of `top_k(n+m)`, which the `@matched` evaluation arm
depends on ("the same retriever, just more results"). Fixing it moved lexical
recall@3 from 0.665 to 0.660: the old number was partly luck of tie order.

### Property graphs, identity, and `MERGE`
A property graph has nodes with *labels* (`:Entity:Class`) and properties
(`name`), and typed, directed relationships (`CALLS`). Cypher's `MERGE` means
"match this whole pattern, create it if absent", and the pattern includes
the **label**. So `MERGE (:Class {name_normalized: "x"})` and
`MERGE (:Entity {name_normalized: "x"})` are two different nodes. Whatever
goes into the `MERGE` pattern *is* the node's identity key.

Supporting pieces:
- **Uniqueness constraint**: the database itself refuses a second node with
  the same key. It protects identity across chunks and concurrent loads,
  which per-chunk grammar constraints can't see.
- **Index**: makes the `MATCH`/`MERGE` lookup fast instead of a scan.
- **`UNWIND` batching**: one query per batch of rows instead of one round
  trip per row.
- **Labels and relationship types can't be query parameters.** They have to
  be written into the query text, so only a closed set of trusted values
  (the schema enums) may ever be interpolated. That's an injection boundary.

**In this project:** identity is `(:Entity {name_normalized})` alone, with the
extracted type added as an extra label. `ensure_graph_constraints()` creates
the uniqueness constraint and a `Chunk` span index; rows are batched per label
/ relation type because those can't be parameters.

### Entity resolution in practice
Deciding that two mentions refer to the same thing. Two families:
- **Deterministic**: equal keys after a fixed transformation (lowercase,
  strip punctuation, canonical IDs). Cheap, explainable, never wrong about
  what it merges, but misses paraphrases.
- **Probabilistic / fuzzy**: similarity scores plus a threshold. Catches more,
  but a wrong merge is *silent*: two different things become one node and
  every answer built on it is quietly corrupted. A duplicate, by contrast, is
  visible and cheap to fix later.

**In this project:** deterministic only, on purpose. The key design lesson came
from a bug: identity was first `(type label, name)`, and the 0.6B model types
the same identifier inconsistently (`Ray` as `Module` in one chunk, `Entity`
in another). That split 78 names across 165 nodes, and relationship endpoints
created under the fallback label were cut off from any chunk, dead ends for
traversal. Fix: take the noisy field (type) out of the key, keep it as extra
information (a label). The live graph was migrated in place after a JSON
backup. **Never put a noisy attribute in an identity key.**

### Graph traversal: paths, distance, ranking
A *path* alternates nodes and relationships: node, edge, node, edge, node.
Its length is the number of edges. A variable-length pattern
`-[:A|B*0..2]-` matches every path of 0 to 2 edges of those types; with no
arrow it ignores direction, so a parent class, a subclass, a caller and a
callee all count as "one hop".

**In this project** (`traversal.py`), one Cypher query:
- **distance** = the shortest path length (`min(length(path))`) from any seed
  entity to any entity mentioned in the candidate chunk. The `MENTIONED_IN`
  steps in and out aren't counted. Distance 0 = the chunk mentions the same
  entity as a seed.
- **support** = how many distinct seed entities reach the chunk.
- **ranking**: distance asc, support desc, then `file_path`/`first` so the
  order is deterministic.
- **seeds are excluded before `LIMIT`**, so the cap counts only new chunks
  (an earlier version filtered after the cap and silently returned fewer).
- **`max_expanded`** protects against hub entities: one entity mentioned
  everywhere (e.g. `vllm`, 50 chunks; `self`, 18) can otherwise pull in
  hundreds of chunks.
- Known cost risk: enumerating every path to take the minimum is fine at
  ~1.5k nodes, unmeasured at full-corpus scale.

### Evaluating retrieval honestly
- **recall@k**: share of questions where some retrieved chunk covers the
  answer (here: covers ≥50% of the ground-truth span). Measures *finding* the
  answer, not answer quality.
- **Ablation**: remove or replace one component, hold everything else fixed,
  measure the difference.
- **Confound**: a second difference between the compared arms that could
  explain the result on its own. The classic one in RAG: the augmented arm
  simply sees *more* context.
- **Budget-matched baseline** (`@matched` here): the same retriever given
  exactly as many chunks as the graph arm got, per question. Isolates *which*
  chunks the graph picks from *how many*. Without it, this project's first
  result (lexical +0.045 from the graph) was really k chunks vs k+50.
- **Random control** (`+random` here): same count of extra chunks, drawn at
  random from the graph. Rules out "everything in the graph is an answer"
  (leakage): the graph was built from answer chunks, so any pick from it
  could look good.
- **Upper bound / distractors**: a graph made only of answer chunks can't
  mislead expansion, so its numbers are optimistic. Loading non-answer
  distractor chunks is the realistic test.
- **Reach vs usefulness**: the graph can reach chunks BM25 never returns at
  any k (zero term overlap). That's *reach*. `@matched` measures
  *usefulness*: are those chunks the answer more often than the retriever's
  next picks?
- **Per-split results**: averages hide structure. Here docs vs code behave
  differently (the graph helps code more; hybrid beats lexical once, on code
  at k=5).

**Standard names** for the two controls: *budget-matched baseline* and
*random-expansion control*; the whole setup is an *ablation study*.

---

## 4. Architecture — where each piece fits

| Module | Job | Depends on |
|---|---|---|
| `chunking/` | Corpus → offset-tracked `Chunk`s: AST per def/class for `.py`, headers for `.md/.rst/.txt`, line windows for everything else; picks `source_type` by extension | nothing |
| `retrieval/lexical/` | Identifier-aware tokenizer + BM25 index (`bm25s`), `search()` | `chunking/`, `ranking.py` |
| `retrieval/semantic/embeddings.py` | Dense embeddings (`all-MiniLM-L6-v2`), `semantic_top_k()` | `chunking/`, `ranking.py` |
| `retrieval/ranking.py` | Shared deterministic `top_k()` | nothing |
| `retrieval/fusion.py` | RRF `fuse()` + `hybrid_top_k()` | lexical + semantic |
| `extraction/schema.py` | Fixed node/relationship vocabulary; grammar source for constrained decoding *and* what the Neo4j loader writes against | nothing — build first |
| `extraction/extractor.py` | Runs Qwen3-0.6B + constrained decoder over each chunk → validated `ExtractionResult`. Separate prompts for code vs text chunks | `schema.py`, `extraction/prompts/`, `Chunk.source_type` |
| `extraction/prompts/` | Code-chunk prompt vs text-chunk prompt | `schema.py` |
| `graph/neo4j_client.py` | Connection/session handling | nothing |
| `graph/loader.py` | Constraint + index setup; writes `Chunk` nodes, `:Entity` nodes (identity = normalized name), `MENTIONED_IN` and relationship edges, batched with `UNWIND` | `neo4j_client.py`, `schema.py`, `extractor.py`'s output shape |
| `graph/traversal.py` | One Cypher query for query-time steps 2-4 above: distance/support ranking, seeds excluded before the cap | `neo4j_client.py`, loaded graph |
| `pipeline/index_pipeline.py` | Offline: corpus → chunk → extract → load graph (BM25 and embeddings are built separately) | `chunking/`, `extraction/`, `graph/` |
| `pipeline/query_pipeline.py` | Runtime: cache → BM25 → graph expand → prompt → answer → cache | `retrieval/lexical/`, `graph/traversal.py`, `cache/` |
| `cache/cache.py` | Disk-backed exact-match query cache | nothing |
| `evaluation/evaluate.py` | recall@k for lexical/semantic/hybrid × plain/+graph/@matched/+random | retrieval, `graph/traversal.py` |
| `tests/load_eval_subset.py` | Extract + load only the ground-truth chunks (+ optional distractors) | `chunking/`, `extraction/`, `graph/` |

---

## 5. What the project showed

**Results** (recall@k, 200 questions, graph = ground-truth chunks only):
- Lexical (BM25) > hybrid > semantic on a codebase: 0.660 / 0.565 / 0.355 at k=3.
- Graph vs budget-matched baseline: ±0.01 for lexical and hybrid, **+0.03 to
  +0.04 for semantic**. The graph recovers identifier-linked chunks that
  embeddings miss; for BM25 it's no better than retrieving more.
- Random control ≈ plain, so the gains come from following edges, not from
  the graph containing only answers.
- Still an upper bound until the distractor run.

**Lessons that transfer to any project:**
- **Valid ≠ correct.** Constrained decoding guarantees structure; correctness
  needs its own checks (validation, grounding, audits of what was extracted).
- **Every comparison needs a control.** A positive number without a
  budget-matched baseline wasn't evidence; here it was mostly extra context.
- **Identity keys must be stable.** A noisy attribute in a key splits
  entities silently.
- **Prefer the visible failure.** A duplicate node beats a wrong merge; a
  skipped chunk beats a loaded half-extraction.
- **Measure before building.** Content-based file-type detection would have
  changed 0 files in this corpus, and naive "does it parse as Python" would
  have misfiled 403 JSON files.
- **Count the blast radius of a change.** Chunk boundaries key the BM25 rows,
  the embeddings and the graph; changing chunking means rebuilding all three.
- **Scope destructive operations.** A blanket `DETACH DELETE` once wiped the
  graph; test data now carries a marker prefix and cleanup deletes only that.
