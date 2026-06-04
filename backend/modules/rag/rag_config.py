"""Chunk config classes"""

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RAGResponse:
    """
    Response object returned by RAG pipeline.
    """

    answer: str
    sources: list[dict]

@dataclass
class ChunkMetadata:
    """
    Metadata container for a RAG chunk.
    """

    source: str
    file: str
    header: str | None
    level: int | None
    article_number: str | None
    chunk_index: int | None = None

    topics: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    related_articles: list[str] = field(default_factory=list)

    chapter: str | None = None
    paragraph: str | None = None
    legal_domain: str | None = None
    article_title: str | None = None

    graph_rag: dict[str, Any] = field(default_factory=dict)
    classic_rag: dict[str, Any] = field(default_factory=dict)

    context_summary: str | None = None


@dataclass
class Chunk:
    """
    Single text chunk used for indexing, retrieval and generation.
    """

    text: str
    metadata: ChunkMetadata
    chunk_id: str | None = None

    def __post_init__(self) -> None:
        if not self.text:
            return

        if not self.chunk_id:
            key = "|".join([
                self.metadata.source or "",
                self.metadata.file or "",
                str(self.metadata.article_number or ""),
                str(self.metadata.header or ""),
                self.text[:400]
            ])

            self.chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, key))


    def to_payload(self) -> dict[str, Any]:
        """
        Converts chunk into a flat dictionary for vector DB storage.

        Returns:
            dict[str, Any]: Serializable representation of chunk.
        """

        return {
            "text": self.text,
            "source": self.metadata.source,
            "file": self.metadata.file,
            "header": self.metadata.header,
            "level": self.metadata.level,
            "article_number": self.metadata.article_number,
            "chunk_index": self.metadata.chunk_index,
            "topics": self.metadata.topics,
            "keywords": self.metadata.keywords,
            "related_articles": self.metadata.related_articles,
            "chapter": self.metadata.chapter,
            "paragraph": self.metadata.paragraph,
            "legal_domain": self.metadata.legal_domain,
            "article_title": self.metadata.article_title,
            "graph_rag": self.metadata.graph_rag,
            "classic_rag": self.metadata.classic_rag,
            "context_summary": self.metadata.context_summary,
        }
