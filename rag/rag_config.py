import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Literal, Any


@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict]

@dataclass
class ChunkMetadata:
    source: str
    file: str
    chunk_type: Literal["structure", "sentence", "window"]
    header: str | None
    level: int | None
    article_number: str | None
    topics: List[str] = field(default_factory=list)

@dataclass
class Chunk:
    """
    Единица данных (чанк) в RAG-пайплайне.

    Используется на всех этапах:
    - разбиение документов (chunking)
    - генерация эмбеддингов
    - загрузка в векторное хранилище

    Attributes:
        text (str): Текст чанка
        metadata (ChunkMetadata): Метаданные
        chunk_id (str): Уникальный ID (генерируется автоматически, если не задан)
    """

    text: str
    metadata: ChunkMetadata
    chunk_id: str | None = None

    def __post_init__(self) -> None:
        if not self.chunk_id:
            key = self.metadata.source + self.text[:512]
            self.chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, key))

    def to_payload(self) -> Dict[str, Any]:
        """
        Преобразует metadata → payload для Qdrant
        """

        return {
            "text": self.text,
            "source": self.metadata.source,
            "file": self.metadata.file,
            "chunk_type": self.metadata.chunk_type,
            "header": self.metadata.header,
            "level": self.metadata.level,
            "article_number": self.metadata.article_number,
            "topics": self.metadata.topics,
        }
