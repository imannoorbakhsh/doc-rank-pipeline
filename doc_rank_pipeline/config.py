from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


@dataclass(frozen=True)
class ChunkingConfig:
    size: int = 120
    overlap: int = 20

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("chunk size must be positive")
        if not 0 <= self.overlap < self.size:
            raise ValueError("chunk overlap must be non-negative and less than size")


@dataclass(frozen=True)
class RetrievalConfig:
    bm25_enabled: bool = True
    dense_enabled: bool = True
    fusion_enabled: bool = True
    reranker_enabled: bool = True
    candidate_limit: int = 20
    rrf_k: int = 60
    embedding_dimensions: int = 256

    def __post_init__(self) -> None:
        if not self.bm25_enabled and not self.dense_enabled:
            raise ValueError("at least one retriever must be enabled")
        if self.fusion_enabled and not (self.bm25_enabled and self.dense_enabled):
            raise ValueError("fusion requires both BM25 and dense retrieval")
        if self.candidate_limit < 1 or self.rrf_k < 1 or self.embedding_dimensions < 8:
            raise ValueError("retrieval limits and embedding dimensions must be positive")


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    chunking: ChunkingConfig
    retrieval: RetrievalConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PipelineConfig":
        return cls(
            name=str(raw["name"]),
            chunking=ChunkingConfig(**raw.get("chunking", {})),
            retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "PipelineConfig":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
