from dataclasses import dataclass, field
from typing import List, Dict, Literal


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
