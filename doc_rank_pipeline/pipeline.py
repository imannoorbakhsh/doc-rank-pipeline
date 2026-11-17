from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .ingestion import chunk_documents, load_documents
from .models import Chunk, ScoredChunk
from .retrieval import BM25Retriever, HashEmbeddingRetriever, LexicalReranker, reciprocal_rank_fusion


class DocumentRankPipeline:
    def __init__(self, config: PipelineConfig, chunks: list[Chunk]) -> None:
        self.config = config
        self.chunks = chunks
        retrieval = config.retrieval
        self.bm25 = BM25Retriever(chunks) if retrieval.bm25_enabled else None
        self.dense = (
            HashEmbeddingRetriever(chunks, retrieval.embedding_dimensions)
            if retrieval.dense_enabled
            else None
        )
        self.reranker = LexicalReranker() if retrieval.reranker_enabled else None

    @classmethod
    def from_corpus(cls, config: PipelineConfig, corpus_path: str | Path) -> "DocumentRankPipeline":
        return cls(config, chunk_documents(load_documents(corpus_path), config.chunking))

    def search(self, query: str, limit: int = 10) -> list[ScoredChunk]:
        retrieval = self.config.retrieval
        candidate_limit = max(limit, retrieval.candidate_limit)
        rankings: list[list[ScoredChunk]] = []
        if self.bm25:
            rankings.append(self.bm25.search(query, candidate_limit))
        if self.dense:
            rankings.append(self.dense.search(query, candidate_limit))
        if len(rankings) == 2 and retrieval.fusion_enabled:
            candidates = reciprocal_rank_fusion(rankings, retrieval.rrf_k, candidate_limit)
        elif rankings:
            candidates = rankings[0]
        else:
            candidates = []
        return self.reranker.rerank(query, candidates, limit) if self.reranker else candidates[:limit]
