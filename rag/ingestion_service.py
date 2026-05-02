from abc import ABC, abstractmethod
from typing import List

from rag.ingestion import IngestionPipeline
from rag.rag_config import Chunk


class BaseIngestionService(ABC):

    @abstractmethod
    def load_chunks(self) -> List[Chunk]:
        raise NotImplementedError


class IngestionService:
    def __init__(self, pipeline: IngestionPipeline):
        self.pipeline = pipeline

    def load_chunks(self) -> List[Chunk]:
        chunks = self.pipeline.run()
        print(f"[Ingestion] Loaded chunks: {len(chunks)}")
        return chunks
    