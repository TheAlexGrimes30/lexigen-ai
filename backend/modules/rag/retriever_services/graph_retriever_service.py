import re
from abc import ABC, abstractmethod
from typing import Optional, Any

from llama_index.core import Document

from backend.modules.rag.retriever_services.hybrid_retriever_service import HybridRetrieverConfig
from backend.modules.rag.retriever_services.retriever_service import SearchResultFactory, MetadataAdapter, \
    TextTokenizer, ChunkAdapter
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

class LlamaIndexMetadataGraphRetriever(BaseGraphRetriever):
    """
    Lightweight GraphRAG retriever based on LlamaIndex Documents metadata.
    """

    def __init__(
            self,
            config: HybridRetrieverConfig,
            chunks: Optional[list[Any]] = None
    ) -> None:
        """
        Initialize metadata graph retriever.
        """

        self.config = config
        self.documents: list[Document] = []
        self.doc_id_to_docs: dict[str, list[int]] = {}
        self.topic_to_doc_ids: dict[str, set[str]] = {}
        self.doc_graph: dict[str, set[str]] = {}

        if chunks:
            self.build(chunks)

    def build(
            self,
            chunks: list[Any]
    ) -> None:
        """
        Build metadata graph from chunks.
        """

        self.documents = []
        self.doc_id_to_docs = {}
        self.topic_to_doc_ids = {}
        self.doc_graph = {}

        for chunk in chunks:
            document = ChunkAdapter.to_llama_document(
                chunk=chunk,
                config=self.config
            )

            if document is None:
                continue

            index = len(self.documents)
            self.documents.append(document)

            metadata = document.metadata or {}

            doc_id = MetadataAdapter.get_document_id(
                metadata=metadata,
                config=self.config
            )

            if doc_id is None:
                continue

            self.doc_id_to_docs.setdefault(doc_id, []).append(index)
            self.doc_graph.setdefault(doc_id, set())

            self._index_topics(
                doc_id=doc_id,
                metadata=metadata
            )

            self._index_relations(
                doc_id=doc_id,
                metadata=metadata
            )

    def _index_topics(
            self,
            *,
            doc_id: str,
            metadata: dict[str, Any]
    ) -> None:
        """
        Index topics as graph concepts.
        """

        topics: list[Any] = []

        for key in self.config.topic_keys:
            direct_topics = metadata.get(key)

            if isinstance(direct_topics, list):
                topics.extend(direct_topics)

        for key in self.config.nested_topic_keys:
            nested = metadata.get(key)

            if isinstance(nested, dict):
                nested_topics = nested.get("topics") or []

                if isinstance(nested_topics, list):
                    topics.extend(nested_topics)

        for topic in topics:
            topic_key = str(topic).lower().strip()

            if topic_key:
                self.topic_to_doc_ids.setdefault(
                    topic_key,
                    set()
                ).add(doc_id)

    def _index_relations(
            self,
            *,
            doc_id: str,
            metadata: dict[str, Any]
    ) -> None:
        """
        Index graph relations from configured graph metadata.
        """

        graph_metadata = metadata.get(self.config.graph_metadata_key)

        if not isinstance(graph_metadata, dict):
            return

        relations = graph_metadata.get("relations") or []

        if not isinstance(relations, list):
            return

        for relation in relations:
            if not isinstance(relation, dict):
                continue

            target_doc_id = self._extract_relation_target(relation)

            if target_doc_id:
                self.doc_graph.setdefault(
                    doc_id,
                    set()
                ).add(target_doc_id)

    def _extract_relation_target(
            self,
            relation: dict[str, Any]
    ) -> Optional[str]:
        """
        Extract relation target from configured relation target keys.
        """

        for key in self.config.relation_target_keys:
            value = relation.get(key)

            normalized = MetadataAdapter.normalize_id(value)

            if normalized is not None:
                return self._extract_document_id_from_text(normalized)

        return None

    @staticmethod
    def _extract_document_id_from_text(
            value: str
    ) -> str:
        """
        Extract document id from common ids like law_article_133_1.

        This is generic syntax parsing, not a domain-specific correction.
        """

        match = re.search(
            r"(?:article|norm|document|doc)[_\s-]*(\d+)(?:[_\.-](\d+))?$",
            value,
            flags=re.IGNORECASE
        )

        if not match:
            return value

        first = match.group(1)
        second = match.group(2)

        if second is not None:
            return f"{first}.{second}"

        return first

    def search(
            self,
            query: str,
            k: int
    ) -> list[SearchResult]:
        """
        Search graph by query concepts and related document ids.
        """

        if not self.documents:
            return []

        query_tokens = set(
            TextTokenizer.tokenize(query)
        )

        doc_scores: dict[str, float] = {}

        for topic, doc_ids in self.topic_to_doc_ids.items():
            topic_tokens = set(
                TextTokenizer.tokenize(topic)
            )

            if not topic_tokens:
                continue

            overlap = len(query_tokens & topic_tokens)

            if overlap <= 0:
                continue

            for doc_id in doc_ids:
                doc_scores[doc_id] = (
                    doc_scores.get(doc_id, 0.0)
                    + overlap
                )

        for doc_id in list(doc_scores.keys()):
            related_doc_ids = self.doc_graph.get(doc_id, set())

            for related_doc_id in related_doc_ids:
                doc_scores[related_doc_id] = (
                    doc_scores.get(related_doc_id, 0.0)
                    + doc_scores[doc_id] * 0.5
                )

        ranked_doc_ids = sorted(
            doc_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        results: list[SearchResult] = []

        for doc_id, doc_score in ranked_doc_ids:
            doc_indexes = self.doc_id_to_docs.get(doc_id, [])

            for doc_index in doc_indexes:
                document = self.documents[doc_index]

                payload = MetadataAdapter.normalize_payload(
                    payload=dict(document.metadata or {}),
                    config=self.config
                )

                payload["retrieval_doc_id"] = doc_id
                payload["retrieval_source"] = "graph"

                results.append(
                    SearchResultFactory.create(
                        id=f"graph:{doc_id}:{doc_index}",
                        text=document.text,
                        score=float(doc_score),
                        payload=payload
                    )
                )

                if len(results) >= k:
                    return results

        return results
