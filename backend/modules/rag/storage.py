from dataclasses import dataclass
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, Filter, PointStruct, VectorParams


@dataclass
class VectorStore:
    """
    Thin wrapper over Qdrant vector database.
    """

    client: QdrantClient
    collection_name: str
    vector_size: int
    distance: Distance = Distance.COSINE

    def ensure_collection(self) -> None:
        """
        Ensure that Qdrant collection exists.
        """

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance,
                ),
            )

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
        batch_size: int = 64,
    ) -> None:
        """
        Insert or update embeddings in Qdrant.
        """

        if not ids or not vectors:
            raise ValueError("[VectorStore] Empty ids or vectors")

        points: list[PointStruct] = []

        for point_id, vector, payload in zip(ids, vectors, payloads):
            if not vector or len(vector) != self.vector_size:
                continue

            points.append(
                PointStruct(
                    id=str(point_id),
                    vector=vector,
                    payload=self._normalize_payload(payload),
                )
            )

        if not points:
            raise ValueError("[VectorStore] No valid points to upsert")

        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i:i + batch_size],
                wait=True,
            )

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        query_filter: Optional[Filter] = None,
    ) -> list[Any]:
        """
        Perform similarity search in Qdrant collection.
        """

        if not query_vector:
            return []

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )

        return result.points if hasattr(result, "points") else result

    def delete_collection(self) -> None:
        """
        Delete Qdrant collection if it exists.
        """

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

    def _normalize_payload(self, payload: Optional[dict]) -> dict:
        """
        Normalize metadata payload before storing in Qdrant.

        Important: do not drop rich metadata. GraphRAG and metadata-aware
        retrieval depend on these fields.
        """

        payload = dict(payload or {})

        normalized = {
            "text": payload.get("text", "") or "",
            "source": payload.get("source"),
            "file": payload.get("file"),
            "header": payload.get("header"),
            "level": payload.get("level"),
            "article_number": payload.get("article_number"),
            "chunk_index": payload.get("chunk_index"),
            "topics": payload.get("topics") or [],
            "keywords": payload.get("keywords") or [],
            "related_articles": payload.get("related_articles") or [],
            "chapter": payload.get("chapter"),
            "paragraph": payload.get("paragraph"),
            "legal_domain": payload.get("legal_domain"),
            "article_title": payload.get("article_title"),
            "tree_rag": payload.get("tree_rag") or {},
            "graph_rag": payload.get("graph_rag") or {},
            "classic_rag": payload.get("classic_rag") or {},
            "context_summary": payload.get("context_summary"),
        }

        for key, value in payload.items():
            normalized.setdefault(key, value)

        return normalized
