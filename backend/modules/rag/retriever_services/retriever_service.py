from abc import abstractmethod, ABC
from typing import Optional, Any

from backend.modules.rag.search_result_service import SearchResult


class BaseRetriever(ABC):
    """
    Abstract application retriever.
    """

    @abstractmethod
    def retrieve(
            self,
            query: str,
            top_k: int = 10
    ) -> list[SearchResult]:
        """
        Retrieve relevant chunks.
        """

        raise NotImplementedError

class SearchResultFactory:
    """
    Factory for SearchResult objects.
    """

    @staticmethod
    def create(
            *,
            id: Optional[str],
            text: str,
            score: float,
            payload: dict[str, Any]
    ) -> SearchResult:
        """
        Create SearchResult regardless of constructor shape.
        """

        try:
            return SearchResult(
                id=id,
                text=text,
                score=score,
                payload=payload
            )
        except TypeError:
            result = SearchResult.__new__(SearchResult)
            result.id = id
            result.text = text
            result.score = score
            result.payload = payload
            return result

