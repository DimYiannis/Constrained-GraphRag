<div align="center">

# Constrained GraphRAG

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-managed-DE5FE9?logo=uv&logoColor=white)
![bm25s](https://img.shields.io/badge/bm25s-lexical%20retrieval-orange)
![Outlines](https://img.shields.io/badge/Outlines-constrained%20decoding-9146FF)

**Finding the right ~2000 characters out of 28,246 chunks of the vLLM codebase via BM25.**

</div>

---

## 🧩 What is this project?

Constrained GraphRAG combines lexical-first retrieval (BM25) with **schema-constrained knowledge graph extraction** over the **vLLM 0.10.1** codebase (~2,800 files, docs + Python source).

The core idea: entity/relationship extraction from source chunks is done by **Qwen3-0.6B** — a small model, not a frontier one — made trustworthy for this job through **grammar-constrained decoding** via [**Outlines**](https://github.com/dottxt-ai/outlines). The extraction schema (`extraction/schema.py`) is a Pydantic model handed directly to Outlines, which compiles it into a finite state machine and masks the model's logits at every generated token so only schema-conforming output is ever sampled — a mathematical guarantee, not a "the model was told to behave" guarantee. Malformed or duplicate-entity output (`"Acme Corp"` vs `"Acme Corporation"`) becomes structurally impossible rather than something to catch after the fact — which is what makes a 0.6B model viable for structured extraction at all. The resulting graph is stored in **Neo4j** and powers multi-hop traversal on top of retrieval: BM25 finds the chunks that lexically match a query, the graph then pulls in related chunks that never shared a single term with it.

A corpus is chunked into focused, offset-tracked spans, indexed with BM25 over an identifier-aware tokenizer, and searchable from a CLI — no embeddings, no vector index, no LLM in the loop yet.

```
uv run python -m src search "enable lora" --k 5
```

```
1. data/raw/vllm-0.10.1/tests/lora/test_tokenizer_group.py [2141:2747] score=5.07
2. data/raw/vllm-0.10.1/tests/lora/test_llama_tp.py [8720:9022] score=4.78
3. data/raw/vllm-0.10.1/vllm/transformers_utils/tokenizer_group.py [691:1287] score=4.74
```

## 🗂 Project Structure

```
constrained-graphrag/
├── src/
│   ├── __main__.py             # Fire CLI — index, search
│   ├── chunking/
│   │   ├── chunk_corpus.py     # corpus walking, file dispatch by extension
│   │   ├── ast_chunker.py      # AST-based chunking for .py (function/class-level)
│   │   ├── plain_chunker.py    # header-based chunking for markdown, line fallback
│   │   └── spans.py            # shared span-splitting + Chunk dataclass
│   └── retrieval/
│       └── lexical/
│           ├── tokenizer.py    # identifier-aware tokenizer (subtokens, stopwords)
│           └── indexer.py      # Index build/save/load, top-k search (bm25s-backed)
├── data/                        # gitignored, populated locally, never committed
├── pyproject.toml
├── uv.lock
└── README.md
```

Click a folder to see what's inside it:

<details>
<summary>📁 <strong>src/</strong></summary>

The CLI entry point (`__main__.py`, Fire-based) plus every module the pipeline is built from so far — chunking and lexical retrieval.

</details>

<details>
<summary>📁 <strong>src/chunking/</strong></summary>

Structure-aware chunkers with exact character offsets: AST-based splitting for Python (function/class-level), header-based sectioning for markdown, and a line-window fallback for everything else. Every produced span passes through a shared cap-enforcing splitter so no chunk exceeds `max_chunk_size`.

</details>

<details>
<summary>📁 <strong>src/retrieval/lexical/</strong></summary>

BM25 retrieval: an identifier-aware tokenizer (splits `enable_lora` into whole and subtoken forms), and the index build/persist/search logic backed by `bm25s`.

</details>

<details>
<summary>📁 <strong>data/</strong></summary>

Holds the corpus, under `data/raw/<corpus-name>/`. Gitignored — nothing under it is committed, and no path is hard-coded anywhere in the project; every input/output location is a CLI argument.

</details>

---




## 📎 Resources

- [graphrag.com](https://graphrag.com/) — general GraphRAG background.
- Robertson & Zaramba, *The Probabilistic Relevance Framework: BM25 and Beyond* — BM25 scoring, `k1`/`b`.
- [bm25s documentation](https://github.com/xhluca/bm25s) — the BM25 library used here.
- [Python `ast` module docs](https://docs.python.org/3/library/ast.html) — used for structure-aware Python chunking.
- [uv docs](https://docs.astral.sh/uv/) — dependency/project management.
- [How to Build Type-Safe, Schema-Constrained, and Function-Driven LLM Pipelines Using Outlines and Pydantic](https://www.marktechpost.com/2026/03/14/how-to-build-type-safe-schema-constrained-and-function-driven-llm-pipelines-using-outlines-and-pydantic/) — Outlines + Pydantic structured generation.
- [Structured Output (JSON) — LoRAX Docs](https://loraexchange.ai/guides/structured_output/) — constrained JSON generation background.
- [Outlines — structured JSON/regex/Pydantic LLM generation](https://hermes-agent.nousresearch.com/docs/user-guide/skills/optional/mlops/mlops-inference-outlines) — how Outlines' FSM-based constraining works.

