from abc import ABC, abstractmethod

from backend.modules.rag.search_result_service import SearchResult


class BaseGraphRetriever(ABC):
    """
    Abstract interface for GraphRAG retrievers.
    """

    @abstractmethod
    def search(
            self,
            query: str,
            k: int
    ) -> list[SearchResult]:
        """
        Search by graph context.
        """

        raise NotImplementedError