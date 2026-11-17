from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from .models import Chunk, ScoredChunk

_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(tokenize(chunk.text)) for chunk in self.chunks]
        self.lengths = [sum(frequencies.values()) for frequencies in self.term_frequencies]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        document_frequency: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            document_frequency.update(frequencies.keys())
        total = len(self.chunks)
        self.idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, limit: int) -> list[ScoredChunk]:
        query_terms = tokenize(query)
        scored: list[ScoredChunk] = []
        for chunk, frequencies, length in zip(self.chunks, self.term_frequencies, self.lengths):
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / self.average_length)
                score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
            if score:
                scored.append(ScoredChunk(chunk, score))
        return _rank(scored, limit)


class HashEmbeddingRetriever:
    def __init__(self, chunks: Sequence[Chunk], dimensions: int) -> None:
        self.chunks = list(chunks)
        self.dimensions = dimensions
        self.vectors = [self._embed(chunk.text) for chunk in self.chunks]

    def _embed(self, text: str) -> dict[int, float]:
        counts: defaultdict[int, float] = defaultdict(float)
        for token in tokenize(text):
            index = _stable_hash(token) % self.dimensions
            counts[index] += 1.0 if _stable_hash(token + "#") % 2 else -1.0
        magnitude = math.sqrt(sum(value * value for value in counts.values()))
        return {index: value / magnitude for index, value in counts.items()} if magnitude else {}

    def search(self, query: str, limit: int) -> list[ScoredChunk]:
        query_vector = self._embed(query)
        scored = [
            ScoredChunk(chunk, sum(query_vector.get(index, 0.0) * value for index, value in vector.items()))
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        return _rank([item for item in scored if item.score > 0.0], limit)


def reciprocal_rank_fusion(rankings: Iterable[Sequence[ScoredChunk]], k: int, limit: int) -> list[ScoredChunk]:
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, start=1):
            scores[item.chunk.chunk_id] += 1.0 / (k + position)
            chunks[item.chunk.chunk_id] = item.chunk
    return _rank([ScoredChunk(chunks[chunk_id], score) for chunk_id, score in scores.items()], limit)


class LexicalReranker:
    def rerank(self, query: str, candidates: Sequence[ScoredChunk], limit: int) -> list[ScoredChunk]:
        query_terms = set(tokenize(query))
        rescored = []
        for item in candidates:
            chunk_terms = set(tokenize(item.chunk.text))
            coverage = len(query_terms & chunk_terms) / len(query_terms) if query_terms else 0.0
            rescored.append(ScoredChunk(item.chunk, item.score + coverage))
        return _rank(rescored, limit)


def _rank(items: Sequence[ScoredChunk], limit: int) -> list[ScoredChunk]:
    return sorted(items, key=lambda item: (-item.score, item.chunk.chunk_id))[:limit]


def _stable_hash(value: str) -> int:
    result = 2166136261
    for character in value.encode("utf-8"):
        result ^= character
        result = (result * 16777619) & 0xFFFFFFFF
    return result
