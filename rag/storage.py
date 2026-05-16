from dataclasses import dataclass
from typing import Optional, Any, List, Dict

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, PointStruct, VectorParams, Distance


@dataclass
class VectorStore:
    """
    Абстракция над векторным хранилищем (Qdrant).

    Отвечает за:
    - создание коллекции
    - вставку (upsert) векторов
    - поиск по векторам
    - удаление коллекции

    Позволяет изолировать бизнес-логику RAG от конкретной реализации базы.
    """

    client: QdrantClient
    collection_name: str
    vector_size: int
    distance: Distance = Distance.COSINE

    def ensure_collection(self) -> None:
        """
        Проверяет наличие коллекции и создаёт её при отсутствии.

        Поведение:
            - если коллекция уже существует → ничего не делает
            - если нет → создаёт новую с заданными параметрами
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
            ids: List[str],
            vectors: List[List[float]],
            payloads: List[dict],
            batch_size: int = 64
    ) -> None:
        """
        Добавляет или обновляет точки в векторном хранилище.

        Вход:
            ids (List[str]):
                Список уникальных идентификаторов точек

            vectors (List[List[float]]):
                Список векторов (эмбеддингов)
                Размер каждого вектора должен совпадать с vector_size

            payloads (List[dict]):
                Список метаданных для каждой точки
                (например: text, source, дополнительные поля)

        Выход:
            None
        """

        if not ids or not vectors:
            raise ValueError("[VectorStore] Empty ids or vectors")

        points: List[PointStruct] = []

        for i, v, p in zip(ids, vectors, payloads):

            if not v or len(v) != self.vector_size:
                continue

            points.append(
                PointStruct(
                    id=str(i),
                    vector=v,
                    payload=self._normalize_payload(p),
                )
            )

        if not points:
            raise ValueError("[VectorStore] No valid points to upsert")

        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]

            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,
            )


    def search(
            self,
            query_vector: List[float],
            limit: int = 10,
            query_filter: Optional[Filter] = None,
    ) -> List[Any]:
        """
        Выполняет поиск ближайших векторов в коллекции.

        Вход:
            query_vector (List[float]):
                Вектор запроса (embedding)

            limit (int):
                Максимальное количество возвращаемых результатов

            query_filter (Optional[Filter]):
                Фильтр Qdrant для ограничения поиска
                (например: по source, тегам и т.д.)

        Выход:
            List[Any]:
                Список найденных объектов (ScoredPoint из Qdrant),
                содержащих:
                - id
                - score (релевантность)
                - payload (метаданные)

        Примечание:
            Возвращаемый тип оставлен как Any,
            чтобы не привязываться жёстко к Qdrant API.
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
        Удаляет коллекцию из Qdrant (если существует).

        Используется:
            - при очистке базы
            - при переинициализации индекса
        """

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

    def _normalize_payload(self, p: Optional[Dict]) -> Dict:

        p = p or {}

        return {
            "text": p.get("text", "") or "",
            "source": p.get("source"),
            "file": p.get("file"),
            "header": p.get("header"),
            "level": p.get("level"),
            "article_number": p.get("article_number"),
            "topics": p.get("topics") or [],
        }
