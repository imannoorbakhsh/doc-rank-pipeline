from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    text: str
    ordinal: int


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    category: str
    relevant_chunk_ids: frozenset[str]


@dataclass(frozen=True)
class EvaluationResult:
    query_id: str
    category: str
    metrics: Mapping[str, float]
