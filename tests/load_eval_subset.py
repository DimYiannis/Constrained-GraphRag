# PYTHONPATH=/Users/yiannis/Developer/graphrag caffeinate -i uv run python tests/load_eval_subset.py
"""
    one-off: extract + load into Neo4j only the chunks that overlap a
    evaluation/test_queries.json ground-truth span. Same chunking function/
    max_chunk_size as the persisted BM25 index, so chunk boundaries match
    exactly and traversal.expand_chunks can find these as seeds.

    resumable: skips any chunk already present as a Chunk node (idempotent
    MERGE means re-running is safe either way, this just avoids re-paying
    extraction cost for chunks already done).
"""
import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/Users/yiannis/Developer/graphrag/.env")

from src.chunking.chunk_corpus import chunk, read_text
from src.extraction import extractor
from src.graph import loader, neo4j_client

DATA_DIR = Path("data/raw/vllm-0.10.1")
MAX_CHUNK_SIZE = 2000
MAX_NEW_TOKENS = 768

data = json.loads(Path("evaluation/test_queries.json").read_text())
gt_by_file: dict[str, list[tuple[int, int]]] = {}
for q in data:
    for s in q["sources"]:
        gt_by_file.setdefault(s["file_path"], []).append(
            (s["first_character_index"], s["last_character_index"])
        )

wanted = []
for fp, spans in gt_by_file.items():
    text = read_text(DATA_DIR / fp)
    for piece in chunk(text, fp, MAX_CHUNK_SIZE):
        if any(piece.first < last and first < piece.last for first, last in spans):
            wanted.append(piece)

print(f"{len(wanted)} chunks to extract across {len(gt_by_file)} files", flush=True)

driver = neo4j_client.get_driver()

records, _, _ = neo4j_client.run_query(
    driver, "MATCH (c:Chunk) RETURN c.file_path AS file_path, c.first AS first, c.last AS last"
)
done = {(r["file_path"], r["first"], r["last"]) for r in records}
print(f"{len(done)} chunks already loaded, skipping those", flush=True)

model = extractor.load_model()
generator = extractor.build_generator(model)

failures = []
try:
    for i, piece in enumerate(wanted, start=1):
        if (piece.file_path, piece.first, piece.last) in done:
            continue
        try:
            result = extractor.extract(generator, piece, max_new_tokens=MAX_NEW_TOKENS)
            loader.load_chunk(driver, piece, result)
            print(f"[{i}/{len(wanted)}] {piece.file_path} [{piece.first}:{piece.last}] "
                  f"-> {len(result.entities)} entities, {len(result.relationships)} rels", flush=True)
        except Exception as exc:  # noqa: BLE001 - one bad chunk shouldn't kill the batch
            failures.append((piece.file_path, piece.first, piece.last, str(exc)))
            print(f"[{i}/{len(wanted)}] {piece.file_path} [{piece.first}:{piece.last}] "
                  f"FAILED: {exc}", flush=True)
finally:
    driver.close()

print(f"done - {len(failures)} failures", flush=True)
for f in failures:
    print("  FAILED:", f, flush=True)
