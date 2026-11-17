# TODO

This repository intentionally stops at an approximately 60% working milestone. Before using results to make a quality claim, the human owner must:

- Replace `evals/golden_placeholder.json` with a human-reviewed, versioned golden question set sourced from the real corpus.
- Tune chunk sizes, overlap, BM25/dense weights, fusion settings, embedding dimensions/model, candidate limits, and reranker behavior on a development split.
- Run error analysis by query category, including false positives, missed relevant chunks, and duplicate or overly broad chunks.
- Write the tradeoff recommendation covering quality, latency, memory, and Raspberry Pi 4 operational cost.
- Record the end-to-end demo and note hardware, Python version, corpus version, and runtime measurements.
- Add a real corpus ingestion command and licensing/source metadata for the selected public document set.
- Consider replacing the deterministic hash embedder with a compact, locally cached embedding model only after measuring its Pi resource cost.
- Expand regression fixtures beyond the placeholder set and review baseline changes deliberately.
