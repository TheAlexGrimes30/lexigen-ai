import re
from abc import abstractmethod, ABC
from typing import Optional, Any

from llama_index.core import Document

from backend.modules.rag.retriever_services.hybrid_retriever_service import HybridRetrieverConfig
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

class MetadataAdapter:
    """
    Domain-neutral metadata adapter.

    It does not fix specific article numbers and does not contain
    labor-law-specific rules. Domain-specific corrections should live in
    ingestion or evaluation, not in the retriever.
    """

    @staticmethod
    def metadata_to_dict(
            metadata: Any
    ) -> dict[str, Any]:
        """
        Convert metadata object to plain dict.
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
    def normalize_id(
            value: Any
    ) -> Optional[str]:
        """
        Normalize a legal document identifier to string without domain hacks.
        """

        if value is None:
            return None

        text = str(value).strip()

        if not text:
            return None

        if text.endswith(".0"):
            text = text[:-2]

        return text

    @classmethod
    def get_document_id(
            cls,
            metadata: dict[str, Any],
            config: HybridRetrieverConfig
    ) -> Optional[str]:
        """
        Return document/norm/article id from configured metadata keys.
        """

        for key in config.document_id_keys:
            value = metadata.get(key)

            normalized = cls.normalize_id(value)

            if normalized is not None:
                return normalized

        return None

    @classmethod
    def get_header(
            cls,
            metadata: dict[str, Any],
            config: HybridRetrieverConfig
    ) -> Optional[str]:
        """
        Return human-readable header/title from configured metadata keys.
        """

        for key in config.header_keys:
            value = metadata.get(key)

            if value is None:
                continue

            text = str(value).strip()

            if text:
                return text

        return None

    @classmethod
    def normalize_payload(
            cls,
            payload: dict[str, Any],
            config: HybridRetrieverConfig
    ) -> dict[str, Any]:
        """
        Normalize generic payload fields used by retrieval and debugging.
        """

        payload = dict(payload or {})

        document_id = cls.get_document_id(
            metadata=payload,
            config=config
        )

        if document_id is not None:
            payload["retrieval_doc_id"] = document_id

            if "article_number" in payload:
                payload["article_number"] = document_id

        header = cls.get_header(
            metadata=payload,
            config=config
        )

        if header is not None:
            payload["header"] = header

        return payload

class ChunkAdapter:
    """
    Converts project chunks into plain fields and LlamaIndex documents.
    """

    @staticmethod
    def extract_text(
            chunk: Any
    ) -> str:
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
            chunk: Any,
            config: HybridRetrieverConfig
    ) -> dict[str, Any]:
        """
        Extract normalized metadata from chunk.
        """

        metadata_raw = (
            getattr(chunk, "metadata", None)
            or getattr(chunk, "payload", None)
            or {}
        )

        metadata = MetadataAdapter.metadata_to_dict(metadata_raw)

        return MetadataAdapter.normalize_payload(
            payload=metadata,
            config=config
        )

    @classmethod
    def to_llama_document(
            cls,
            chunk: Any,
            config: HybridRetrieverConfig
    ) -> Optional[Document]:
        """
        Convert chunk to LlamaIndex Document.
        """

        text = cls.extract_text(chunk)
        metadata = cls.extract_metadata(
            chunk=chunk,
            config=config
        )

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