from abc import abstractmethod, ABC

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