# Running the Project

Full command sequence to go from a fresh clone to actually querying the
system. `make` targets are the easy path; the raw commands underneath each
one are there if you want to see (or need to tweak) what's actually running.

---

## 1. Install dependencies

```bash
make install
# or: uv sync
```

## 2. Set up `.env`

Create a `.env` file at the repo root (gitignored, never committed):

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<pick a real password>
```

Loaded automatically by `src/__main__.py` (`load_dotenv()`) — no manual
`export` needed once this file exists.

## 3. Start Neo4j

```bash
make neo4j-up
# or, first time:
#   docker run -d --name graphrag-neo4j \
#     -p 7474:7474 -p 7687:7687 \
#     -e NEO4J_AUTH=neo4j/<same password as .env> \
#     neo4j:latest
# or, container already exists:
#   docker start graphrag-neo4j
```

First run downloads the `neo4j:latest` image and creates the container;
later runs just start the existing one. Wait for it to actually finish
booting before continuing:

```bash
make neo4j-logs
# or: docker logs -f graphrag-neo4j
```
(watch for a `Started.` line, then `Ctrl+C` — the container keeps running)

To stop it later:

```bash
make neo4j-down
# or: docker stop graphrag-neo4j
```

Full setup details and troubleshooting (stuck Docker daemon, stuck
installer, etc.) are in `docker-setup.md`.

## 4. Get a corpus

Place one under `data/raw/<corpus-name>/` — this project was built and
tested against the vLLM 0.10.1 source tree. `data/` is gitignored except
`data/raw/`, which is tracked.

## 5. Build the BM25 index

```bash
make index
# or: uv run python -m src index --data_directory data/raw/vllm-0.10.1
```

Chunks every file under the corpus and builds/persists the BM25 index to
`data/processed/`. Takes a few seconds for the full vLLM corpus
(~28,000 chunks).

## 6. Search (lexical only, fully working)

```bash
make search QUERY="enable lora"
# or: uv run python -m src search "enable lora" --k 5
```

Returns ranked `(file_path, span, score)` results — no LLM, no graph, just
BM25 over the index built in step 5.

## 7. Load chunks into the graph

Not yet exposed as a `make`/CLI command — run directly:

```bash
uv run python -c "
from pathlib import Path
from src.pipeline import index_pipeline

processed = index_pipeline.run(Path('data/raw/vllm-0.10.1'), limit=20)
print(f'processed {processed} chunks')
"
```

`limit` defaults to 20, not the full corpus — extraction measured
~15-30s/chunk on a single machine, so the full ~28,000-chunk corpus is
days of compute. Raise `limit` (or pass `None`) deliberately, expecting the
runtime that implies.

## 8. Answer a question (retrieve → graph expand → answer)

```bash
make answer QUERY="how does enable_lora work"
# or: uv run python -m src answer "how does enable_lora work" --data_directory data/raw/vllm-0.10.1
```

**Important:** `--data_directory` here must be the *same* corpus root used
in step 5 (index) and step 7 (graph load) — `file_path` values only match
across BM25 results and graph `Chunk` nodes if computed relative to the
same root. Answers will only reflect content whose chunks were actually
loaded into the graph in step 7 — with the default `limit=20`, that's a
small slice of the corpus, not the whole thing.

---

## Everyday commands, once set up

```bash
make neo4j-up          # or: docker start graphrag-neo4j
make search QUERY="..."   # or: uv run python -m src search "..." --k 5
make answer QUERY="..."   # or: uv run python -m src answer "..." --data_directory <corpus root>
make test               # or: uv run pytest tests/ -v
make clean               # or: find . -type d -name "__pycache__" -exec rm -rf {} + && rm -rf .pytest_cache
```
