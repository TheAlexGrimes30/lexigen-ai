import re
from abc import abstractmethod, ABC
from typing import Optional, Any

from llama_index.core import Document

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

class ChunkAdapter:
    """
    Converts project chunks into plain fields and LlamaIndex documents.
    """

    @staticmethod
    def metadata_to_dict(metadata: Any) -> dict[str, Any]:
        """
        Convert metadata object to dict.
        """

        if metadata is None:
            return {}

        if isinstance(metadata, dict):
            return dict(metadata)

        if hasattr(metadata, "model_dump"):
            return dict(metadata.model_dump())

        if hasattr(metadata, "dict"):
            return dict(metadata.dict())

        if hasattr(metadata, "__dict__"):
            return {
                key: value
                for key, value in metadata.__dict__.items()
                if not key.startswith("_")
            }

        return {}

    @staticmethod
    def extract_text(chunk: Any) -> str:
        """
        Extract text from chunk.
        """

        text = (
                getattr(chunk, "text", None)
                or getattr(chunk, "content", None)
                or getattr(chunk, "page_content", None)
                or ""
        )

        return str(text)

    @classmethod
    def extract_metadata(
            cls,
            chunk: Any
    ) -> dict[str, Any]:
        """
        Extract normalized metadata from chunk.
        """

        metadata_raw = (
                getattr(chunk, "metadata", None)
                or getattr(chunk, "payload", None)
                or {}
        )

        metadata = cls.metadata_to_dict(metadata_raw)

        header = metadata.get("header")

        if header is not None:
            metadata["header"] = str(header).strip()

        return metadata

    @classmethod
    def to_llama_document(
            cls,
            chunk: Any
    ) -> Optional[Document]:
        """
        Convert chunk to LlamaIndex Document.
        """

        text = cls.extract_text(chunk)
        metadata = cls.extract_metadata(chunk)

        if not text.strip():
            return None

        return Document(
            text=text,
            metadata=metadata
        )

class TextTokenizer:
    """
    Simple Russian-friendly tokenizer for BM25.
    """

    TOKEN_PATTERN = re.compile(r"[а-яА-ЯёЁa-zA-Z0-9_.]+")

    @classmethod
    def tokenize(
            cls,
            text: str
    ) -> list[str]:
        """
        Tokenize text.
        """

        return [
            token.lower()
            for token in cls.TOKEN_PATTERN.findall(text or "")
        ]