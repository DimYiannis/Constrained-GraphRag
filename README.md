<div align="center">

# Constrained GraphRAG

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-managed-DE5FE9?logo=uv&logoColor=white)
![bm25s](https://img.shields.io/badge/bm25s-lexical%20retrieval-orange)
![Outlines](https://img.shields.io/badge/Outlines-constrained%20decoding-9146FF)
![sentence-transformers](https://img.shields.io/badge/sentence--transformers-semantic%20retrieval-2E8B57)
![RRF](https://img.shields.io/badge/RRF-hybrid%20fusion-1E90FF)

**Finding the right ~2000 characters out of 28,246 chunks of the vLLM codebase via BM25.**

![answering a real query end-to-end, then again served instantly from cache](./assets/answer_cache_demo.gif)

*The full pipeline retrieve → graph expand → prompt → answer, on a real question, run twice: once cold, once served from `cache/cache.py`.*

📖 [**RUNNING.md**](./RUNNING.md): full setup + command reference · 🐳 [**docker-setup.md**](./docker-setup.md): Neo4j via Docker

**Contents:** [What is this](#-what-is-this-project) · [Architecture](#-architecture) · [Workflow](#-workflow) · [Built On](#-built-on) · [Project Structure](#-project-structure) · [Extraction](#-extraction--constrained-decoding) · [Closed Taxonomy](#-closed-taxonomy-vs-open-labeling) · [Entity Resolution](#-entity-resolution) · [**Results**](#-results) · [Limitations](#️-limitations) · [Challenges](#-challenges-faced) · [Future Improvements](#-future-improvements) · [Resources](#-resources)

</div>

---

## 🧩 What is this project?

Constrained GraphRAG combines lexical-first retrieval (BM25) with **schema-constrained knowledge graph extraction** over the **vLLM 0.10.1** codebase (~2,800 files: docs, Python source, plus C++/CUDA kernels and config files).

The core idea, as a flow: each chunk is run through **Qwen3-0.6B**, constrained by [**Outlines**](https://github.com/dottxt-ai/outlines) so its output always matches a fixed schema.

1. The LLM decides which entities relate and how, Outlines just guarantees that decision comes out structurally valid.
2. Extracted entities and relationships are loaded into **Neo4j** as a graph, connecting chunks through the entities they share.
3. BM25 search at query time, BM25 finds the chunks that lexically match.
4. Graph traversal: the graph is traversed outward from them to pull in related chunks, linked through shared or connected entities, that BM25 didn't rank in its top-k (some share no term with the query at all).

Mechanics of *how* the constraining actually works are in [Extraction](#-extraction--constrained-decoding) below.

**Why this over a naive lexical-only RAG** (the shape of a prior BM25-only project): plain lexical retrieval has no concept of "entity" or "relationship", only term overlap with the query. The identifier-aware tokenizer already splits `TokenizerGroup` and `tokenizer_group` into shared tokens, so BM25 handles spelling variants. What it can't do is follow structure: the class a matched chunk inherits from, the function it calls, or the doc page that names it may share no term with the query at all. The graph makes those reachable. Entities from every chunk are merged into one node per normalized name at load time (`loader.py`), so *every* chunk mentioning an entity is reachable from every other one, and from there along the extracted `CALLS`/`INHERITS_FROM`/`REFERENCES`/… edges. Paraphrases and abbreviations (`Acme Corp` vs `Acme Corporation`) are not merged; see [Entity Resolution](#-entity-resolution).

A corpus is chunked into focused, offset-tracked spans and indexed with BM25 over an identifier-aware tokenizer. The `search` command is pure BM25 (no embeddings, no LLM); `answer` adds graph expansion and Qwen3-0.6B generation; dense embeddings and RRF fusion are used by the evaluation.

```
uv run python -m src search "enable lora" --k 5
```

```
1. tests/lora/test_tokenizer_group.py [2141:2747] score=5.07
2. tests/lora/test_llama_tp.py [8720:9022] score=4.78
3. vllm/transformers_utils/tokenizer_group.py [691:1287] score=4.74
4. tests/lora/test_llama_tp.py [9025:9360] score=4.73
5. tests/lora/test_transformers_model.py [2718:3340] score=4.65
```

---

## 🏗 Architecture

**Offline — indexing** (three separate steps: `src index` builds BM25, the snippet in RUNNING.md builds the embeddings, and `pipeline/index_pipeline.py` / `tests/load_eval_subset.py` run extraction and load Neo4j):

```mermaid
flowchart LR
    corpus[("vLLM corpus<br/>~2,800 files")]

    subgraph chunking["Chunking"]
        direction TB
        ast["AST chunker<br/>(.py; line windows for<br/>code/config files)"]
        plain["Plain chunker<br/>(.md / .rst / .txt by headers)"]
    end

    bm25["BM25 Index<br/>(tokenizer + bm25s)"]
    semb["Sentence-Transformers<br/>(embeddings)"]
    embmatrix[("Embeddings Matrix<br/>(.npy)")]
    qwen["Qwen3-0.6B + Outlines<br/>(constrained extraction)"]
    neodb[("Neo4j<br/>Chunk nodes +<br/>entities + relationships")]

    corpus --> ast
    corpus --> plain
    ast --> bm25
    plain --> bm25
    ast --> semb
    plain --> semb
    semb --> embmatrix
    ast --> qwen
    plain --> qwen
    qwen --> neodb
```

**Online — answering a query** (`answer` → `pipeline/query_pipeline.py`, BM25 seeds only):

```mermaid
flowchart LR
    user(["User"])
    cache[("Query cache")]
    lex[("BM25 Index")]
    seeds["Seed Chunks<br/>(top-5)"]
    neodb[("Neo4j<br/>traverse 0-2 hops")]
    merged["Seed + Expanded<br/>Chunks (≤50 expanded)"]
    llm["Qwen3-0.6B"]

    user -- "Question" --> cache
    cache -- "hit" --> user
    cache -- "miss" --> lex
    lex --> seeds

    seeds -- "1. find seeds' entities<br/>2. expand outward" --> neodb
    neodb -- "3. new chunks" --> merged
    seeds --> merged

    merged -- "Prompt" --> llm
    llm -- "Answer (stored in cache)" --> user
```

The evaluation (`evaluation/evaluate.py`) runs the same graph expansion on top of three retrievers: BM25, dense embeddings (`all-MiniLM-L6-v2`), and hybrid (RRF fusion of both, `c=60`). The `answer` command uses BM25 only, the strongest of the three on this corpus (see [Results](#-results)).

---

## 🔄 Workflow

 The two diagrams above show data flowing between modules, this is the actual division of labor behind that flow, offline and online pieces both:

**The corpus is chunked (`chunking/`) → each chunk runs through `extractor.py`, where the LM creates entity-relationships *within* that one chunk → `loader.py`'s `MERGE` incidentally connects *across* chunks by reusing shared entity names → *(at query time)* BM25 (`retrieval/lexical/`) finds lexically-matching seed chunks → `traversal.py`'s `MATCH` *reads* the already-existing connections, expanding outward from those seeds to pull in chunks BM25 never lexically matched.**

---

## 🔗 Built On

Two earlier projects each contributed one core technique reused here, adapted rather than copied wholesale:

- **[RAG Against the Machine](https://github.com/DimYiannis/RAG-Against-the-Machine)**: the BM25 lexical retriever. `retrieval/lexical/tokenizer.py` and `indexer.py` are a direct port of that project's identifier-aware tokenizer and `bm25s`-backed index, adapted to this project's chunk metadata and package layout.
- **[call me maybe](https://github.com/DimYiannis/call_me_maybe)**: a function-calling engine that constrains a Qwen3-0.6B model's output token-by-token via a hand-rolled state machine over the model's vocabulary, so it can only ever emit valid, schema-conforming JSON. That project's core idea — grammar-constrained decoding making a 0.6B model reliable for structured generation — is exactly the mechanism `extraction/schema.py` is designed around here, this time via the [Outlines](https://github.com/dottxt-ai/outlines) library instead of a hand-rolled decoder.

---

## 🗂 Project Structure

Click a folder, then click a file inside it, to see what it does:

<details>
<summary>📁 <strong>src/</strong></summary>

<details>
<summary>📄 <code>__main__.py</code></summary>

The CLI entry point. Fire turns each method on `RagCLI` into a command: `index` chunks a corpus and builds the BM25 index, `search` prints ranked BM25 results, and `answer` runs the full retrieve → graph expand → generate pipeline (BM25 seeds only) and prints the sources and the answer.

</details>

<details>
<summary>📁 <strong>chunking/</strong></summary>

<details>
<summary>📄 <code>chunk_corpus.py</code></summary>

Walks a corpus directory, decides each file's chunking strategy and `source_type` ("code" vs "text") by extension, dispatches to the right chunker below.

</details>

<details>
<summary>📄 <code>ast_chunker.py</code></summary>

Chunks Python files by parsing the AST; each top-level function/class becomes its own chunk, decorators included, and a class over `max_chunk_size` is split per method. Code between definitions is chunked separately. Also defines `chunk_lines`, the line-window chunker used when a Python file fails to parse and for every non-Python, non-markdown file (C++/CUDA, YAML, JSON, shell, …).

</details>

<details>
<summary>📄 <code>plain_chunker.py</code></summary>

Chunks markdown, `.rst` and `.txt` by ATX headers (each `#`…`######` starts a new section), merging sections under 600 characters into the previous one.

</details>

<details>
<summary>📄 <code>spans.py</code></summary>

The `Chunk` dataclass, and the shared span-splitting logic (cuts at a blank line, then any newline, then mid-line) every chunker funnels through to enforce `max_chunk_size`.

</details>

</details>

<details>
<summary>📁 <strong>retrieval/</strong></summary>

<details>
<summary>📄 <code>lexical/tokenizer.py</code></summary>

Turns text into BM25 search terms: lowercases, and splits identifiers into both whole and subtoken forms (`enable_lora` → `enable_lora`, `enable`, `lora`) so a query can match either way.

</details>

<details>
<summary>📄 <code>lexical/indexer.py</code></summary>

Builds/saves/loads the BM25 index (backed by `bm25s`), and `search()` — turns a query into ranked, tie-broken chunk results.

</details>

<details>
<summary>📄 <code>semantic/embeddings.py</code></summary>

Dense retrieval: embeds every chunk with `sentence-transformers` (`all-MiniLM-L6-v2`) into an L2-normalized matrix, persisted alongside the BM25 index. `semantic_top_k()` ranks by cosine similarity — a plain dot product, since normalized vectors make that equivalent to cosine similarity without recomputing norms per query.

</details>

<details>
<summary>📄 <code>fusion.py</code></summary>

Reciprocal Rank Fusion of a lexical ranking and a semantic ranking: `fuse()` scores each chunk by `Σ 1/(c + rank)` across both rankings — rank-based, not score-based, since BM25 scores and cosine similarities live on completely different scales and directly combining them would let one silently dominate. `hybrid_top_k()` wraps `lexical.search()` + `semantic_top_k()` + `fuse()` into one call, pulling a wider candidate pool (`FUSION_CANDIDATES`, default 100) from each retriever before fusing down to the final `k` — otherwise a chunk ranked low by one retriever but high by the other would never get the chance to surface.

</details>

<details>
<summary>📄 <code>ranking.py</code></summary>

`top_k()`, the top-k selection shared by BM25 and dense retrieval. Ties go to the lower chunk id across the whole score array, so `top_k(n)` is always a prefix of `top_k(n+m)`, which the @matched evaluation arm relies on.

</details>

</details>

<details>
<summary>📁 <strong>extraction/</strong></summary>

`schema.py`, `extractor.py`, `prompts/code_prompt.py` / `text_prompt.py` — the grammar, the constrained generator, and the two extraction prompts. Covered in depth in [Extraction & Constrained Decoding](#-extraction--constrained-decoding).

</details>

<details>
<summary>📁 <strong>graph/</strong></summary>

<details>
<summary>📄 <code>neo4j_client.py</code></summary>

Connection handling: builds a driver from `NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD` (URI and user default to `bolt://localhost:7687`/`neo4j`; the password must come from the environment and is never hardcoded), verifies it can actually reach the database, a thin wrapper for running a Cypher query.

</details>

<details>
<summary>📄 <code>loader.py</code></summary>

Writes one chunk's `ExtractionResult` into Neo4j: the `Chunk` node, each entity, a `MENTIONED_IN` edge from each listed entity to the chunk, and the extracted relationship edges, batched with `UNWIND` (one query per type label / relation type, not one per row). `ensure_graph_constraints()` adds the uniqueness constraint on `(:Entity {name_normalized})` and the `Chunk` span index. The entity-merge logic (the actual entity-resolution mechanism) is covered in [Entity Resolution](#-entity-resolution).

</details>

<details>
<summary>📄 <code>traversal.py</code></summary>

Takes a retriever's top-k results as seed chunks (lexical, semantic, or hybrid — this function doesn't care which), walks 0 to `hops` (default 2) entity-to-entity edges outward from the entities mentioned in them (0 = another chunk mentioning the same entity), and returns the *other* chunks reachable that way, ranked closest-first then by how many seed entities reach them, capped at `max_expanded` after the seeds are excluded.

</details>

</details>

<details>
<summary>📁 <strong>pipeline/</strong></summary>

<details>
<summary>📄 <code>index_pipeline.py</code></summary>

Offline orchestration: walks a corpus, chunks every file, runs each chunk through `extractor.py`, loads the result into Neo4j via `loader.py`. `limit` defaults to 20 rather than the full ~28,246-chunk corpus: extraction runs ~60-90s/chunk, so a full run is weeks of compute, not something to kick off by default.

</details>

<details>
<summary>📄 <code>query_pipeline.py</code></summary>

Runtime orchestration: BM25 search for seed chunks → `traversal.py`'s graph expansion outward from them → build the final prompt from both → generate an answer. Checks `cache/cache.py`'s persisted cache first and returns early on a hit, skipping retrieval/expansion/generation entirely; writes the result back to the cache on a miss.

</details>

</details>

<details>
<summary>📁 <strong>cache/</strong></summary>

`cache.py` — persistent, disk-backed exact-match cache keyed on `(query, k, hops, max_new_tokens)`; disk-backed since each CLI call is its own process. Its exact-match-only limitation is covered in [Challenges Faced](#-challenges-faced).

</details>

</details>

<details>
<summary>📁 <strong>evaluation/</strong></summary>

`evaluate.py` (recall@k for lexical/semantic/hybrid, selected with `--modes`, default lexical; each scored plain, +graph, @matched and +random, printed as one table per k with docs/code splits and a Δ vs @matched column) and `test_queries.json` (the 200-question ground truth). Methodology and numbers in [Results](#-results).

</details>

<details>
<summary>📁 <strong>tests/</strong></summary>

<details>
<summary>📄 <code>test_*.py</code></summary>

Real pytest tests (`make test`). `test_extraction_quality.py` asserts that relationship subjects/targets from *live* model output match the identifier regex (`schema.py`); `test_cache.py` covers the cache round-trip across a reload, key-field misses, and corrupt-file recovery; `test_fusion.py` / `test_ranking.py` / `test_tokenizer.py` cover RRF's rank-only scoring, deterministic top-k tie-breaking, and identifier splitting; `test_traversal.py` builds an isolated marker-prefixed subgraph through `loader.py` in a live Neo4j and checks hop distances, ordering, and the cap (skips if Neo4j is down).

</details>

<details>
<summary>📄 <code>check_chunks.py</code> / <code>check_extraction.py</code> / <code>check_pipeline.py</code></summary>

Standalone debug scripts (not pytest) for eyeballing one stage in isolation — chunking output, a single extraction call, or the offline index pipeline end-to-end on 2 chunks of `vllm/lora` (writes to Neo4j) — against real data, without running the whole CLI.

</details>

<details>
<summary>📄 <code>check_er.py</code></summary>

Verifies entity-resolution normalization actually merges cosmetic name variants into one graph node. Deliberately scoped and non-destructive: test data carries a marker prefix guaranteed never to collide with real corpus data, and cleanup only ever deletes nodes matching that prefix — never a blanket wipe (a blanket `MATCH (n) DETACH DELETE n` here once cost hours of re-extraction to recover from).

</details>

<details>
<summary>📄 <code>load_eval_subset.py</code></summary>

One-off, resumable script: extracts + loads into Neo4j only the chunks that overlap a `test_queries.json` ground-truth span (303, not the full ~28,246-chunk corpus; 295 loaded, 8 failed, see [Challenges Faced](#-challenges-faced)) — enough to run a real graph evaluation without weeks of extraction. `--distractors N` also loads N randomly sampled non-answer chunks (seeded, deterministic) so the graph isn't made of answer chunks only. Skips chunks already present on re-run.

</details>

</details>

<details>
<summary>📁 <strong>data/</strong></summary>

<details>
<summary>📄 <code>raw/&lt;corpus-name&gt;/</code></summary>

Holds the corpus.

</details>

<details>
<summary>📄 <code>processed/</code></summary>

The persisted BM25 index (`index.pkl` + `bm25s/`) and the semantic embeddings matrix (`embeddings.npy`). The index is built by `index`, the embeddings by the snippet in RUNNING.md step 7. `search`/`answer`/`evaluate` load the index; only `evaluate` in semantic or hybrid mode loads the embeddings.

</details>

<details>
<summary>📄 <code>cache/</code></summary>

`cache.py`'s persisted query cache (see `src/cache/`) — survives across CLI invocations since each one is its own process.

</details>

<details>
<summary>📄 <code>backups/</code></summary>

JSON graph snapshots taken before migrations that rewrite the graph, e.g. `graph-2026-09-23-pre-entity-merge.json`, taken before entity identity moved to `:Entity {name_normalized}`.

</details>

</details>

<details>
<summary>📁 <strong>assets/</strong></summary>

<details>
<summary>📄 <code>eval_demo.gif</code> / <code>answer_cache_demo.gif</code></summary>

The two demo GIFs in this README: `answer_cache_demo.gif` at the top (the full pipeline answering a real question twice, cold vs. cache hit) and `eval_demo.gif` in [Results](#-results) (the lexical/semantic/hybrid recall@k comparison). Both are rendered with `agg` from `asciinema` recordings (`.cast` files, kept locally and gitignored).

</details>

<details>
<summary>📄 <code>record_answer_cache_demo.sh</code></summary>

The script the `answer_cache_demo.gif` recording actually runs (kept locally, gitignored): calls `answer` twice with the same query, times each with `date`/`bc`, prints the real elapsed seconds and speedup — makes the cache's effect visible on screen rather than relying on the GIF's own timing.

</details>

</details>

<details>
<summary>📄 Root files</summary>

`Makefile` (setup/index/answer/test shortcuts), [`RUNNING.md`](./RUNNING.md) (setup + command reference), [`docker-setup.md`](./docker-setup.md) (Neo4j via Docker), [`Theory.md`](./Theory.md) (the concepts behind the pipeline), [`todo.md`](./todo.md) (fixed and open items).

</details>

---

## 🧠 Extraction & Constrained Decoding

The extraction schema (`extraction/schema.py`) is a Pydantic model (`ExtractionResult`) handed directly to Outlines. Outlines compiles that schema into a **finite state machine**, and at every single token the model generates, masks every token that would leave a valid path through that FSM down to probability zero before sampling. That's what makes a 0.6B model viable here at all: the reliability gap that would normally require a frontier model is closed by making structurally invalid output unsampleable, rather than by making the model smarter. The one gap is truncation: if generation hits `max_new_tokens` mid-JSON, the output is valid so far but incomplete. `extract()` still runs `ExtractionResult.model_validate_json`, and a chunk that fails validation is skipped (never loaded into the graph), not retried. That happened to 8 chunks, see [Challenges Faced](#-challenges-faced).

Two things worth being precise about scope-wise:

- This constrains *structure and type* (`node_type`/`relation` are enum-restricted fields, `name`/`subject`/`target` are regex-restricted to an identifier shape), it does **not** constrain *which* real-world thing a name refers to. Two chunks extracting the same real entity under different names (`"Acme Corp"` vs `"Acme Corporation"`) is a separate, unsolved problem, see [Entity Resolution](#-entity-resolution) below.
- `CHUNK` and `MENTIONED_IN` are deliberately excluded from what the model is even allowed to emit (see `ExtractableNodeType`/`ExtractableRelationType` in `schema.py`), `Chunk` nodes already exist before extraction runs, and `MENTIONED_IN` is structural (an entity extracted *from* a chunk is trivially mentioned in it), added automatically by `loader.py` for every entity the model lists for that chunk rather than spending the model's constrained generation budget on it. Entities that appear only as a relationship endpoint get no `MENTIONED_IN`.

---

## 🏷 Closed Taxonomy vs. Open Labeling

**Designing a closed relation-type taxonomy, instead of open-ended extraction.** The default approach in most GraphRAG tutorials including [Microsoft's original GraphRAG implementation](https://microsoft.github.io/graphrag/index/default_dataflow/), whose extraction prompt asks for a free-text `relationship_description` rather than a fixed type, is to let the model freely choose relationship labels from context `"calls"`, `"invokes"`, `"is called by"`, `"depends on"`. At small scale this looks harmless. At the scale needed for a usable knowledge graph, it becomes label proliferation: dozens of near-duplicate relation strings fragmenting what should be one queryable edge type, with no clean way back.

### Fixes

- Clustering/Deduplication process could be implemented but it can be lossy and adds a whole extra pipeline stage.

- The alternative, restricting up front, risks losing genuinely useful nuance if the schema is too coarse. Resolved by defining a small, fixed enum of relation types (`CALLS`, `IMPORTS`, `INHERITS_FROM`, `DEFINED_IN`, `RELATES_TO`, `REFERENCES`, see `ExtractableRelationType` in `schema.py`) and enforcing them at generation time via the same FSM-based constrained decoding used throughout this project: the model is only ever able to emit a token sequence resolving to one of the valid types, trading some expressiveness for guaranteed schema consistency. The right tradeoff for a system meant to support reliable multi-hop traversal, less so for open-ended exploratory tagging.

Each type earns its place by doing one specific, well-defined job — except one, deliberately. These definitions are what the extraction prompts ask for; the grammar itself only guarantees the relation is one of the six.

- **`CALLS`** — one function invokes another. Applies only when the code shows an actual call (or a doc chunk explicitly says X calls Y), not "these two functions seem related."
- **`IMPORTS`** — a module imports another module. A literal import statement, not a vague dependency.
- **`INHERITS_FROM`** — a class inherits from another class. Structural, unambiguous — nothing else can mean this.
- **`DEFINED_IN`** — a function or class is defined within a module. Containment, not association.
- **`REFERENCES`** — a chunk names a specific function, class or module defined elsewhere without calling it. Mainly the deliberate cross-modal link (a doc chunk naming code, connecting documentation to code), though code chunks can emit it too.
- **`RELATES_TO`** — "these two entities are related somehow, but not in a way any of the specific types capture." Every other type above is precise *because* this one exists to absorb what doesn't fit without it, ambiguous cases would pressure the specific types to loosen their own definitions instead.

*What I'd do differently:* design the enum with an explicit versioning/extension process from the start, rather than treating it as fixed. A closed schema solves label proliferation, but a genuinely new relationship type the initial design didn't anticipate has nowhere to go except a catch-all like `RELATES_TO`, which just relocates the fuzziness instead of removing it.

- Better: periodically review catch-all usage as a signal for when the enum itself needs a deliberate, reviewed addition — schema evolution as a governed process, not a binary choice between fully open and fully frozen.

---

## 🪪 Entity Resolution

**The closed taxonomy above solves *type* consistency. It says nothing about *identity*.** `NodeType`/`RelationType` only guarantee that two independent extractions are *allowed* to agree a thing is a `Function` or a `Class` — they don't guarantee two mentions of the same real-world thing end up as the same graph node.

**Our problem, concretely:** `graph/loader.py` merges entities on a normalized name: `MERGE (e:Entity {name_normalized})`, where the name is lowercased with everything but `[a-z0-9]` stripped, and the extracted type (`Function`, `Class`, …) is added as an extra label. Two chunks extracting `"TokenizerGroup"` and `"tokenizer_group"` collapse into one shared node, with a `MENTIONED_IN` edge from it to each of the two chunks — the mechanism that makes cross-chunk graph connections happen at all. Identity is deliberately the name alone, not `(label, name)`: the 0.6B model types the same identifier inconsistently (`Ray` as a `Module` in one chunk, a generic `Entity` in another), and keying on the label had silently split 78 names across 165 nodes before this was fixed and the graph migrated. What normalization can't do: `"Acme Corp"` vs `"Acme Corporation"` — two mentions of the same real entity, spelled differently, still become two separate nodes.

[Aakash's writeup on entity resolution](https://www.aakashx.com/blog/knowledge-architecture-ontologies-entity-resolution-graphs/#do-not-hide-ambiguity) deterministic matching (exact ID/key equality) vs. probabilistic/fuzzy matching (similarity + confidence scoring) -> "probabilistic inference should not silently become authoritative master data." Avoid automatical merge of probable similar entities (uncertain inference). If merge is wrong the mistake is silent and harder to catch than a duplicate.

---

## 📊 Results

![lexical vs semantic vs hybrid recall@k, live in the terminal](./assets/eval_demo.gif)

*Live recall@k across all three retrieval modes **lexical** (BM25), **semantic** (`sentence-transformers` dense embeddings), and **hybrid** (Reciprocal Rank Fusion of both), each scored plain, +graph, @matched and +random, overall and split by docs/code. Same numbers as the table below.*

Recall@k on `evaluation/test_queries.json` (200 questions, reused from a sibling project's dataset over the same vLLM 0.10.1 corpus, see `evaluation/evaluate.py`). A hit requires a retrieved chunk to cover at least 50% of the ground-truth answer span. Each retrieval mode is scored four ways:

- **plain** — top-k chunks.
- **+graph** — top-k seeds plus up to k graph-expanded chunks.
- **@matched** — no graph, just a bigger k from the **same** retriever: `lexical@matched` takes more BM25 results, `semantic@matched` more embedding results, `hybrid@matched` more fused results. If `lexical+graph` gave a question 3 seeds + 2 expanded chunks, `lexical@matched` gets BM25's top 5. Both arms get the same chunk budget on every question (`lexical@matched` can get fewer only when BM25 has fewer chunks with any matching term), so the only difference is *which* chunks fill the extra slots: the graph's picks or the retriever's next-ranked ones. This is the fair baseline for +graph.

  It doesn't assume the retriever could reach the graph's chunks, and often it can't: BM25 only returns chunks that share a term with the query, however large k gets, while the graph reaches chunks linked to a seed through entities and relationships the model extracted. That reach is the graph's unique contribution. @matched asks whether the chunks it reaches are the answer more often than what the retriever would have returned next.
- **+random** — top-k seeds plus as many chunks as +graph added, drawn at random from the graph's chunks. A leakage control, since almost every chunk in the graph overlaps a ground-truth answer span.

| k | lexical | +graph | @matched | +random | semantic | +graph | @matched | +random | hybrid | +graph | @matched | +random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 0.660 | 0.675 | 0.680 | 0.660 | 0.355 | 0.395 | 0.365 | 0.355 | 0.565 | 0.585 | 0.585 | 0.565 |
| 5 | 0.705 | 0.720 | 0.715 | 0.705 | 0.400 | 0.450 | 0.420 | 0.400 | 0.650 | 0.680 | 0.675 | 0.650 |
| 10 | 0.785 | 0.795 | 0.785 | 0.790 | 0.515 | 0.585 | 0.545 | 0.520 | 0.725 | 0.755 | 0.755 | 0.725 |

Graph expansion added 1.9 / 3.3 / 6.9 chunks per question for lexical at k = 3 / 5 / 10 (semantic 1.2 / 2.3 / 5.5, hybrid 1.7 / 3.1 / 6.6).

Not the naive "hybrid wins" or "graph wins" story:

- **Lexical is the strongest single retriever here, by a clear margin.** This corpus is a codebase, ground truth hinges on exact identifiers and function names, exactly what BM25's term matching is built for.
- **Semantic alone is meaningfully weaker** (0.355-0.515). It finds *conceptually* related content rather than exact-name matches, for the query `"enable lora"`, semantic surfaces `docs/features/lora.md` and LoRA-handling code in `worker/model_runner.py`, while lexical surfaces `tests/lora/test_tokenizer_group.py` and `transformers_utils/tokenizer_group.py` — genuinely different, both reasonable, but this eval's precise identifier-anchored ground truth rewards lexical's style more.
- **Hybrid lands between the two, closer to lexical than to semantic — and never beats lexical alone overall.** (On the code split it edges ahead once, at k=5: 0.626 vs 0.616.) RRF fusion pulls lexical's strong ranking down by averaging in semantic's weaker one; on a corpus this identifier-precise, fusing in a weaker retriever costs more than it adds.
- **For lexical and hybrid, the graph is no better than retrieving more.** +graph vs @matched is within ±0.01 at every k (lexical even loses at k=3). The graph does reach chunks BM25 can't, but on these questions they aren't the answer more often than BM25's next-ranked results. An earlier version of this table showed lexical +0.045 from the graph; it compared k chunks against k + up to 50 expanded chunks, so most of that was extra budget, not the graph.
- **Semantic is where the graph earns its place:** +0.03 to +0.04 over @matched at every k, a bit more on code (+0.04) than docs (+0.02 to +0.04). Embedding similarity misses identifier-anchored chunks; following shared entities from the seeds recovers some of them.
- **Leakage control:** +random is within 0.005 of plain, so random picks from a graph of (almost only) answer chunks don't explain the graph arm's hits. What this can't show is how well expansion holds up when non-answer chunks compete for the same slots — that needs `load_eval_subset.py --distractors N`, not yet run.

<details>
<summary>Per-split breakdown (docs vs code)</summary>

**docs (n=101)**

| k | lexical | +graph | @matched | semantic | +graph | @matched | hybrid | +graph | @matched |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 0.772 | 0.772 | 0.782 | 0.455 | 0.485 | 0.465 | 0.594 | 0.614 | 0.624 |
| 5 | 0.792 | 0.792 | 0.802 | 0.495 | 0.525 | 0.505 | 0.673 | 0.693 | 0.703 |
| 10 | 0.851 | 0.851 | 0.851 | 0.564 | 0.634 | 0.594 | 0.743 | 0.772 | 0.782 |

**code (n=99)**

| k | lexical | +graph | @matched | semantic | +graph | @matched | hybrid | +graph | @matched |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 0.545 | 0.576 | 0.576 | 0.253 | 0.303 | 0.263 | 0.535 | 0.556 | 0.545 |
| 5 | 0.616 | 0.646 | 0.626 | 0.303 | 0.374 | 0.333 | 0.626 | 0.667 | 0.646 |
| 10 | 0.717 | 0.737 | 0.717 | 0.465 | 0.535 | 0.495 | 0.707 | 0.737 | 0.727 |

On docs the graph never beats @matched for lexical or hybrid; on code it edges ahead (up to +0.02 for lexical, +0.02 for hybrid). Semantic's gap vs. lexical is proportionally wider on code than docs (k=3: code's semantic score is under half of lexical's, docs' is closer to 60%) — semantic similarity has less to latch onto in code identifiers (`kv_cache_coordinator`, `compressed_tensors`) than in prose, which has more natural-language structure the embedding model can actually use.

</details>

---

## ⚠️ Limitations

- **The graph only covers 295 of the corpus's 28,246 chunks** (of the 303 ground-truth chunks, 8 permanently failed extraction on truncation, see Challenges Faced; the graph also holds 4 leftover chunks from early pipeline test runs). Extraction runs one chunk at a time through Qwen3-0.6B (~60-90s/chunk observed). A full-corpus run is weeks of compute, not something to do by default. The graph was instead ingested only over the chunks that `evaluation/test_queries.json`'s ground-truth answers actually live in, enough to run a real evaluation but not a full production-scale graph. Lexical search, by contrast, runs over the entire corpus. Because almost every chunk in the graph overlaps a ground-truth answer span, graph numbers here are an upper bound on what a full-corpus graph would show; `load_eval_subset.py --distractors 300` (~5-7h) is the planned test of that.

- **Type labels are mostly uninformative.** 1,039 of 1,451 entity nodes (72%) carry only the base `:Entity` label. The model *chose* `Entity` for 956 of them; only 83 exist purely as relationship endpoints. Constrained decoding guarantees the label is *one of* the allowed types, not that it's the *right* one. Three causes:
  - **Schema gap:** most identifiers in code (`num_dp`, `device`, `model_input.input_tokens`) are variables, parameters or attributes. None of `Function`/`Class`/`Module`/`Concept` fits them, so `Entity` is the honest answer.
  - **Under-specified prompts:** both prompts list the five types without defining them or marking `Entity` as a last resort, and the text prompt's worked example labels the config flag `enable_lora` as a `Function`.
  - **Genuine misses:** e.g. `FusedMoEActivationFormat` extracted as `Entity`, not `Class`.

  Retrieval is unaffected, since traversal ignores labels. It does limit how readable the graph is, and rules out type-aware traversal (e.g. `INHERITS_FROM` only between `Class` nodes).

- **Junk entities.** The name regex constrains identifier *shape*, so `self`, `with`, `a`, `K` all pass as entity names. `self` is the third-largest hub in the graph (mentioned in 18 chunks), linking otherwise unrelated chunks at distance 0.

---

## 🧗 Challenges Faced

**A 0.6B model doesn't automatically produce clean, real identifiers just because the JSON around them is valid.** Early testing surfaced two distinct failure modes, fixed at two different layers:

- **Prompt-level fix:** the model would sometimes name an entity after the schema's own type vocabulary — literally `"Function"`, `"Class"`, `"Module"` as an entity's *name*, copying the words right next to where `name` was being defined in the prompt. Fixed by extending both extraction prompts (`code_prompt.py`/`text_prompt.py`) with an explicit worked example and a direct instruction: the name is a real identifier from the text, never the literal type-vocabulary words.
- **Schema-level fix:** relationship `subject`/`target` values kept showing up as entire import statements or sentences instead of clean identifiers — structurally valid per the schema at the time (a plain unconstrained `str`), but semantically useless. Prompt wording alone only partially fixed this. The real fix was adding a regex constraint (`Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.]*$")`) directly to `schema.py`'s `name`/`subject`/`target` fields — Outlines compiles that pattern into the same FSM that already enforces `node_type`/`relation`, so a full import statement became *structurally unreachable* to generate, not just discouraged by prompt text.

**One edge case the regex fix doesn't close:** it constrains *shape*, not *truth*. The model can still fabricate a string that looks identifier-shaped but isn't a real symbol anywhere in the source — observed case: `from vllm.utils import LRUCache` got mashed into `"from_vllm.utils.lrucache"` as a relationship target, which satisfies the regex (only letters/underscores/dots) while being a fabrication. A regex can only ever constrain what a string *looks like*; verifying it's a *real* symbol from the actual chunk would need a fundamentally different mechanism.

**The query cache only ever matches identically.** `cache/cache.py` keys on the literal `(query, k, hops, max_new_tokens)` tuple — `"enable lora"` and `"Enable LoRA"` are two totally different cache entries, even though a person reads them as the same question. A cache that catches near-paraphrases needs a **semantic cache** instead — embed the query, compare similarity against past queries.

**A small fraction of chunks never finish extraction, even with a bumped token budget.** Loading the eval ground-truth subset (303 chunks), 8 permanently failed (295 loaded) with `EOF while parsing a string` — the model was still mid-entity when it hit `max_new_tokens` (raised to 768 for this run, up from the default 512), so Outlines' FSM produced valid-so-far-but-incomplete JSON rather than a parseable `ExtractionResult`. Almost always entity-dense code files (attention kernels, quantization backends) generating more entities/relationships than the budget anticipated. Accepted as a known error rate rather than chased further — the fix (a much larger budget, or splitting the chunk itself) trades extraction cost for a rare failure mode that doesn't materially affect graph coverage at this scale.

   ---

## 🔭 Future Improvements

- **A `Variable` node type** (parameters, locals, attributes), so `Entity` means "unknown" rather than "default".
- **Define every node type in both extraction prompts**, mark `Entity` as last resort, and fix the `enable_lora` worked example.
- **Filter non-entities at load time** with a fixed rule (Python keywords, `self`/`cls`, single-letter names). Deterministic, so it stays clear of fuzzy merging.

The first two need a full re-extraction (~60-90s/chunk); the third can prune the existing graph in place. All three need a re-run of the ablation, since changing types and pruning hub entities changes which chunks the graph connects.

---

## 📎 Resources

- [graphrag.com](https://graphrag.com/) — general GraphRAG background.
- [Microsoft GraphRAG — Indexing Dataflow](https://microsoft.github.io/graphrag/index/default_dataflow/) — the open-ended entity/relationship extraction approach this project's closed taxonomy deliberately departs from.
- [The GraphRAG Manifesto — Neo4j](https://neo4j.com/blog/genai/graphrag-manifesto/) — why graph-augmented retrieval beats naive RAG.
- [Cypher `MERGE` clause](https://neo4j.com/docs/cypher-manual/current/clauses/merge/) — idempotent node/relationship creation, used throughout `graph/loader.py`.
- [Hugging Face — Prompting](https://huggingface.co/docs/transformers/main/en/tasks/prompting) — the grounding-prompt pattern `query_pipeline.py`'s `PROMPT_TEMPLATE` follows.
- [Knowledge Graphs for RAG — DeepLearning.AI](https://www.deeplearning.ai/courses/knowledge-graphs-rag) — course on building/querying knowledge graphs for RAG.
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* — BM25 scoring, `k1`/`b`.
- [bm25s documentation](https://github.com/xhluca/bm25s) — the BM25 library used here.
- [Python `ast` module docs](https://docs.python.org/3/library/ast.html) — used for structure-aware Python chunking.
- [uv docs](https://docs.astral.sh/uv/) — dependency/project management.
- [How to Build Type-Safe, Schema-Constrained, and Function-Driven LLM Pipelines Using Outlines and Pydantic](https://www.marktechpost.com/2026/03/14/how-to-build-type-safe-schema-constrained-and-function-driven-llm-pipelines-using-outlines-and-pydantic/) — Outlines + Pydantic structured generation.
- [Structured Output (JSON) — LoRAX Docs](https://loraexchange.ai/guides/structured_output/) — constrained JSON generation background.
- [Outlines — structured JSON/regex/Pydantic LLM generation](https://hermes-agent.nousresearch.com/docs/user-guide/skills/optional/mlops/mlops-inference-outlines) — how Outlines' FSM-based constraining works.
- [Outlines Model Initialization](https://dottxt-ai.github.io/outlines/main/features/models/transformers/?utm_source=chatgpt.com)
- [Loading models from HF](https://huggingface.co/docs/transformers/en/models?utm_source=chatgpt.com)
- [Tokenizer and Auto classes from HF](https://huggingface.co/docs/transformers/model_doc/auto?utm_source=chatgpt.com)
- [Outlines Generator](https://dottxt-ai.github.io/outlines/main/features/core/generator/?utm_source=chatgpt.com)
- [Knowledge Architecture: Ontologies, Entity Resolution, Graphs](https://www.aakashx.com/blog/knowledge-architecture-ontologies-entity-resolution-graphs/#5-18-graphrag)  frames GraphRAG as an architectural pattern worth reaching for only when relationships materially affect the answer ("the graph provides scope, vectors provide relevance").
- [SentenceTransformers — main docs](https://sbert.net/) — the library itself: quickstart, `SentenceTransformer(model_name)`, `.encode()`.
- [Pretrained Models](https://sbert.net/docs/sentence_transformer/pretrained_models.html) — model comparison table, includes `all-MiniLM-L6-v2` (the one used in `retrieval/semantic/embeddings.py` — fast, 384-dim, 256-token cap).
- [Semantic Search](https://sbert.net/examples/sentence_transformer/applications/semantic-search/README.html) — the actual use case here: embed a corpus, embed a query, rank by cosine similarity — same pattern `semantic_top_k()` implements.
- [`.encode()` API reference](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode) — the exact params in use (`batch_size`, `convert_to_numpy`, `normalize_embeddings`, `show_progress_bar`).
- Cormack, Clarke & Buettcher, [*Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (SIGIR 2009) — the original RRF paper; `RRF_C = 60` in `retrieval/fusion.py` is this paper's own tuned constant, not an arbitrary choice.
- [Reciprocal Rank Fusion — Elastic docs](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion) — a more approachable walkthrough of the same `1/(c + rank)` formula, with worked examples.
