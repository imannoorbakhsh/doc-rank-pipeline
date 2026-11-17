from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .config import ChunkingConfig
from .models import Chunk, SourceDocument
from .retrieval import tokenize


def load_documents(path: str | Path) -> list[SourceDocument]:
    with Path(path).open(encoding="utf-8") as handle:
        raw_documents = json.load(handle)
    return [
        SourceDocument(
            document_id=str(document["document_id"]),
            title=str(document["title"]),
            text=str(document["text"]),
        )
        for document in raw_documents
    ]


def chunk_documents(documents: Iterable[SourceDocument], config: ChunkingConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        words = tokenize(document.text)
        step = config.size - config.overlap
        for ordinal, start in enumerate(range(0, len(words), step)):
            text = " ".join(words[start : start + config.size])
            if not text:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}:{ordinal}",
                    document_id=document.document_id,
                    title=document.title,
                    text=text,
                    ordinal=ordinal,
                )
            )
            if start + config.size >= len(words):
                break
    return chunks


def serialize_chunks(chunks: Iterable[Chunk], path: str | Path) -> None:
    records = [chunk.__dict__ for chunk in chunks]
    Path(path).write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
