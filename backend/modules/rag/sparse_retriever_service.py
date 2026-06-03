from abc import ABC

from backend.modules.rag.search_result_service import SearchResult


class BaseSparseRetriever(ABC):
    """
    Abstract interface for sparse retrievers.
    """

    def search(
            self,
            query: str,
            k: int
    ) -> list[SearchResult]:
        """
        Search by text query.
        """

        raise NotImplementedError