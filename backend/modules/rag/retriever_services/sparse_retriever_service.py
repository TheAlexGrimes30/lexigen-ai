from abc import ABC
from typing import Optional, Any

from llama_index.core import Document
from rank_bm25 import BM25Okapi

from backend.modules.rag.retriever_services.retriever_service import ChunkAdapter, TextTokenizer, SearchResultFactory, \
    MetadataAdapter
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

class BM25SparseRetriever(BaseSparseRetriever):
    """
    BM25 sparse retriever over loaded chunks.
    """

    def __init__(
            self,
            config: HybridRetrieverConfig,
            chunks: Optional[list[Any]] = None
    ) -> None:
        """
        Initialize BM25 retriever.
        """

        self.config = config
        self.documents: list[Document] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: Optional[BM25Okapi] = None

        if chunks:
            self.build(chunks)

    def build(
            self,
            chunks: list[Any]
    ):

        documents: list[Document] = []

        for chunk in chunks:
            document = ChunkAdapter.to_llama_document(
                chunk=chunk,
                config=self.config
            )

            if document is not None:
                documents.append(document)

        self.documents = documents

        self.tokenized_corpus = [
            TextTokenizer.tokenize(document.text)
            for document in documents
        ]

        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def search(
            self,
            query: str,
            k: int
    ) -> list[SearchResult]:
        """
        Search BM25 index.
        """

        if self.bm25 is None:
            return []

        query_tokens = TextTokenizer.tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )[:k]

        results: list[SearchResult] = []

        for index in ranked_indexes:
            score = float(scores[index])

            if score <= 0:
                continue

            document = self.documents[index]

            payload = MetadataAdapter.normalize_payload(
                payload=dict(document.metadata or {}),
                config=self.config
            )

            payload["retrieval_source"] = "bm25"

            result = SearchResultFactory.create(
                id=f"bm25:{index}",
                text=document.text,
                score=score,
                payload=payload
            )

            results.append(result)

        return results
