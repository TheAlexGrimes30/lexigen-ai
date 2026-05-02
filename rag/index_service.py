from abc import ABC, abstractmethod
from typing import List

from rag.rag_config import Chunk
from rag.retriever import Embedder
from rag.storage import VectorStore

class BaseIndexService(ABC):

    @abstractmethod
    def index(self, chunks: List[Chunk]) -> None:
        raise NotImplementedError

class IndexService:
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def index(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return

        texts = [c.text for c in chunks]
        payloads = [c.to_payload() for c in chunks]
        ids = [c.chunk_id for c in chunks]

        vectors = self.embedder.encode_passages(texts)

        self.vector_store.upsert(ids, vectors, payloads)

        print(f"[Index] Indexed: {len(chunks)} chunks")
        