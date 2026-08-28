 src/
    chunking/
      ast_chunker.py      ← old chunking.py: chunk_python() + AST helpers
      plain_chunker.py     ← old chunking.py: chunk_markdown() + chunk_lines()
      (dispatcher/Chunk dataclass — shared, exact file TBD)
                            NOT STARTED. source_type gets assigned here.

    retrieval/
      lexical.py           ✅ DONE — ported from tokenizer.py+indexer.py+retriever.py
      dense.py             stub only, per convention (don't build yet)
      fusion.py            stub only, per convention (don't build yet)

    extraction/
      prompts/              NOT STARTED — separate prompts for code vs text chunks
                            (keyed off source_type from chunking)
      extractor.py          NOT STARTED — no old-project equivalent; runs
                            Qwen3-0.6B + constrained decoding, new code
      schema.py             NOT STARTED — node/relationship type defs,
                            also new (old project had no graph extraction)

    graph/
      neo4j_client.py        NOT STARTED — no old-project equivalent
      loader.py               NOT STARTED — writes extracted triples into Neo4j
      traversal.py             NOT STARTED — Cypher for graph expansion

    pipeline/
      index_pipeline.py     NOT STARTED — offline: corpus -> chunk -> extract
                            -> load graph. Orchestrates chunking+lexical.py+
                            extraction+graph, no old equivalent (old project's
                            __main__.py did something similar for indexing only)
      query_pipeline.py     NOT STARTED — runtime: retrieve -> graph expand
                            -> prompt -> answer. Old retriever.py's top_k +
                            search_dataset logic is the retrieval half of this;
                            graph-expansion half is new.

    cache/
      cache.py              old project's cache.py existed on the
                            semantic-hybrid branch (sqlite query-result cache),
                            absent from current lexical-only main. CLAUDE.md
                            calls this "existing, reused as-is" — worth
                            checking that branch when you get here.

  evaluation/
    test_queries.json        NOT STARTED — old project used RagDataset/
                            moulinette-shaped JSON; this one's schema is
                            yours to define for lexical vs hybrid vs
                            hybrid+graph comparisons
    evaluate.py              NOT STARTED — old evaluator.py is a reference
                            point but scoring changes (adds a graph-expansion
                            arm)