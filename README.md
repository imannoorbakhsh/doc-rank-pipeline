# doc-rank-pipeline

A lightweight, offline document-ranking pipeline with reproducible retrieval evaluation for Raspberry Pi-class hardware.

**Status: ~60%, work in progress**

## Architecture

```text
bundled documents -> chunker -> in-memory index
                              |-> BM25 retriever
query -> configurable retrieval|-> hash embedding retriever -> reciprocal rank fusion
                              |-> lexical reranker
                                           -> ranked chunks
                                                  -> evaluation metrics and comparison report
```

The default implementation uses only the Python standard library and a deterministic hash-vector embedder, so it requires no GPU, model download, or service. The embedding interface is intentionally small so a human can later install and select a compact local sentence-transformer model.

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m doc_rank_pipeline.cli evaluate --config configs/default.json
python -m doc_rank_pipeline.cli compare
pytest
```

The sample corpus is a compact, paraphrased learning corpus based on public Python standard-library concepts. It is included only to prove the pipeline runs end to end; replace it with an appropriately licensed corpus for a real evaluation.

## Configurations

- `configs/default.json`: hybrid BM25 + dense hash embeddings + reranker.
- `configs/dense.json`: dense-only retrieval without a reranker.
- `configs/small_chunks.json`: hybrid retrieval with smaller chunks and reranking.

All retrieval layers are independently switchable through configuration:

- `retrieval.bm25_enabled`
- `retrieval.dense_enabled`
- `retrieval.fusion_enabled`
- `retrieval.reranker_enabled`

## Evaluation

The versioned golden set at `evals/golden_placeholder.json` is deliberately marked as a **placeholder**. Its query identifiers target the sample documents and are sufficient for a smoke test only. Evaluation reports recall@k, MRR, and nDCG overall and for each query category.

`python -m doc_rank_pipeline.cli compare` evaluates the three committed configurations and writes a JSON comparison report under `reports/` when explicitly requested with `--output`.

`python -m doc_rank_pipeline.cli check-regression` compares the current recall@10 with `evals/baseline.json` and exits nonzero if the score decreases by more than 0.01 (one percentage point).

## Resource profile

The defaults keep all data in memory, avoid GPU libraries and network calls, and are intended for small corpora on a Raspberry Pi 4. Larger corpora, real embedding models, and tuning work are intentionally deferred to `TODO.md`.
