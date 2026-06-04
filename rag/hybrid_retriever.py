from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from llama_index.core import Document
from rank_bm25 import BM25Okapi

from rag.dense_retriever import Embedder
from rag.search_result import SearchResult


@dataclass(frozen=True, slots=True)
class HybridRetrieverConfig:
    """
    Configuration for Dense + BM25 + metadata/context GraphRAG retrieval.
    """

    alpha: float = 0.88
    graph_weight: float = 0.025
    metadata_weight: float = 0.04
    pool_multiplier: int = 8
    max_pool_size: int = 80
    min_text_len: int = 40
    max_chunks_per_article: int = 2

    graph_seed_top_n: int = 4
    graph_max_related_per_seed: int = 2
    graph_relation_decay: float = 0.45

    document_id_keys: tuple[str, ...] = (
        "article_number",
        "article",
        "article_id",
        "norm_id",
        "document_id",
        "section_number",
    )

    header_keys: tuple[str, ...] = (
        "header",
        "article_title",
        "title",
        "heading",
        "name",
    )

    topic_keys: tuple[str, ...] = ("topics",)
    keyword_keys: tuple[str, ...] = ("keywords",)
    nested_topic_keys: tuple[str, ...] = ("classic_rag",)
    graph_metadata_key: str = "graph_rag"

    relation_target_keys: tuple[str, ...] = (
        "target",
        "to",
        "article",
        "article_number",
        "document_id",
        "norm_id",
    )

    relation_type_weights: dict[str, float] | None = None

    enable_content_edges: bool = True
    enable_topic_edges: bool = True
    enable_tree_edges: bool = True
    enable_related_article_edges: bool = True

    content_reference_weight: float = 0.95
    tree_edge_weight: float = 0.65
    related_article_weight: float = 0.75
    shared_topic_weight: float = 0.08
    shared_keyword_weight: float = 0.12
    max_topic_edges_per_doc: int = 2

    metadata_header_weight: float = 0.35
    metadata_topic_weight: float = 0.25
    metadata_keyword_weight: float = 0.25
    metadata_domain_weight: float = 0.10
    metadata_related_weight: float = 0.05

    @property
    def bm25_weight(self) -> float:
        """
        BM25 weight derived from alpha for backward compatibility.
        """

        return 1.0 - self.alpha

    def relation_weight(self, relation_type: str | None) -> float:
        """
        Return graph relation weight.
        """

        default_weights = {
            "REFERENCES": 0.9,
            "RELATED": 0.55,
            "EXPLAINS": 0.75,
            "APPLIES_TO": 0.8,
            "PROCEDURE_FOR": 0.85,
            "CONSEQUENCE_OF": 0.8,
            "CONTENT_REFERENCE": self.content_reference_weight,
            "TREE": self.tree_edge_weight,
            "RELATED_ARTICLE": self.related_article_weight,
            "SHARED_TOPIC": self.shared_topic_weight,
            "SHARED_KEYWORD": self.shared_keyword_weight,
        }

        weights = self.relation_type_weights or default_weights

        if relation_type is None:
            return 0.5

        return weights.get(str(relation_type).upper().strip(), 0.5)


class BaseDenseRetriever(ABC):
    """
    Abstract interface for dense retrievers.
    """

    @abstractmethod
    def search(self, query_vec: list[float], k: int) -> list[SearchResult]:
        """
        Search by dense vector.
        """

        raise NotImplementedError


class BaseSparseRetriever(ABC):
    """
    Abstract interface for sparse retrievers.
    """

    @abstractmethod
    def search(self, query: str, k: int) -> list[SearchResult]:
        """
        Search by lexical query.
        """

        raise NotImplementedError


class BaseGraphRetriever(ABC):
    """
    Abstract interface for graph expansion retrievers.
    """

    @abstractmethod
    def expand(self, seed_doc_ids: list[str], k: int) -> list[SearchResult]:
        """
        Expand high-confidence seed documents through graph relations.
        """

        raise NotImplementedError


class BaseRetriever(ABC):
    """
    Abstract application retriever.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
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
        payload: dict[str, Any],
    ) -> SearchResult:
        """
        Create SearchResult with fallback for older constructors.
        """

        try:
            return SearchResult(id=id, text=text, score=score, payload=payload)
        except TypeError:
            result = SearchResult.__new__(SearchResult)
            result.id = id
            result.text = text
            result.score = score
            result.payload = payload
            return result


class MetadataAdapter:
    """
    Domain-neutral metadata adapter for legal corpora.
    """

    @staticmethod
    def to_dict(metadata: Any) -> dict[str, Any]:
        """
        Convert metadata object to a plain dict.
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
            return {k: v for k, v in metadata.__dict__.items() if not k.startswith("_")}
        return {}

    @staticmethod
    def normalize_id(value: Any) -> Optional[str]:
        """
        Normalize legal document id.
        """

        if value is None:
            return None

        text = str(value).strip()
        if not text:
            return None

        if text.endswith(".0"):
            text = text[:-2]

        return MetadataAdapter.extract_document_id_from_text(text)

    @staticmethod
    def as_list(value: Any) -> list[str]:
        """
        Convert scalar/list metadata values to list[str].
        """

        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, (tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]

        text = str(value).strip()
        return [text] if text else []

    @staticmethod
    def extract_document_id_from_text(value: str) -> str:
        """
        Extract generic legal document id from common identifiers.
        """

        match = re.search(
            r"(?:article|norm|document|doc)[_\s-]*(\d+)(?:[_\.-](\d+))?",
            value,
            flags=re.IGNORECASE,
        )

        if not match:
            return value

        first = match.group(1)
        second = match.group(2)

        return f"{first}.{second}" if second else first

    @classmethod
    def get_document_id(cls, metadata: dict[str, Any], config: HybridRetrieverConfig) -> Optional[str]:
        """
        Get article/norm/document id from configured metadata keys.
        """

        for key in config.document_id_keys:
            normalized = cls.normalize_id(metadata.get(key))
            if normalized is not None:
                return normalized
        return None

    @classmethod
    def get_header(cls, metadata: dict[str, Any], config: HybridRetrieverConfig) -> Optional[str]:
        """
        Get header/title from configured metadata keys.
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
    def normalize_payload(cls, payload: dict[str, Any], config: HybridRetrieverConfig) -> dict[str, Any]:
        """
        Normalize generic payload fields used by retrieval and debug output.
        """

        payload = dict(payload or {})

        doc_id = cls.get_document_id(payload, config)
        if doc_id is not None:
            payload["retrieval_doc_id"] = doc_id
            payload["article_number"] = cls.normalize_id(payload.get("article_number")) or doc_id

        header = cls.get_header(payload, config)
        if header is not None:
            payload["header"] = header

        payload["topics"] = cls.as_list(payload.get("topics"))
        payload["keywords"] = cls.as_list(payload.get("keywords"))
        payload["related_articles"] = cls.as_list(payload.get("related_articles"))

        classic_rag = payload.get("classic_rag")
        if isinstance(classic_rag, dict):
            payload["topics"] = sorted(set(payload["topics"] + cls.as_list(classic_rag.get("topics"))))
            payload["keywords"] = sorted(set(payload["keywords"] + cls.as_list(classic_rag.get("keywords"))))

        return payload


class ChunkAdapter:
    """
    Converts project chunks into plain fields and LlamaIndex documents.
    """

    @staticmethod
    def extract_text(chunk: Any) -> str:
        """
        Extract text from chunk.
        """

        text = getattr(chunk, "text", None) or getattr(chunk, "content", None) or getattr(chunk, "page_content", None) or ""
        return str(text)

    @classmethod
    def extract_metadata(cls, chunk: Any, config: HybridRetrieverConfig) -> dict[str, Any]:
        """
        Extract normalized metadata from chunk.
        """

        metadata_raw = getattr(chunk, "metadata", None) or getattr(chunk, "payload", None) or {}
        metadata = MetadataAdapter.to_dict(metadata_raw)
        return MetadataAdapter.normalize_payload(metadata, config)

    @classmethod
    def to_llama_document(cls, chunk: Any, config: HybridRetrieverConfig) -> Optional[Document]:
        """
        Convert project chunk to LlamaIndex Document.
        """

        text = cls.extract_text(chunk)
        metadata = cls.extract_metadata(chunk, config)

        if not text.strip():
            return None

        return Document(text=text, metadata=metadata)


class TextTokenizer:
    """
    Simple Russian-friendly tokenizer for BM25, metadata and graph matching.
    """

    TOKEN_PATTERN = re.compile(r"[а-яА-ЯёЁa-zA-Z0-9_.]+")

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        """
        Tokenize text.
        """

        return [token.lower() for token in cls.TOKEN_PATTERN.findall(text or "") if len(token) > 1]


class QdrantDenseRetriever(BaseDenseRetriever):
    """
    Dense retriever based on Qdrant.
    """

    def __init__(self, vector_store: Any, config: HybridRetrieverConfig) -> None:
        self.vector_store = vector_store
        self.config = config

    def search(self, query_vec: list[float], k: int) -> list[SearchResult]:
        """
        Search Qdrant.
        """

        hits = self.vector_store.search(query_vector=query_vec, limit=k)
        results: list[SearchResult] = []

        for hit in hits:
            result = SearchResult.from_qdrant(hit)

            if not result.text or not result.text.strip():
                continue

            result.payload = MetadataAdapter.normalize_payload(result.payload or {}, self.config)
            result.payload["retrieval_source"] = "dense"
            results.append(result)

        return results


class BM25SparseRetriever(BaseSparseRetriever):
    """
    BM25 sparse retriever over loaded chunks.
    """

    def __init__(self, config: HybridRetrieverConfig, chunks: Optional[list[Any]] = None) -> None:
        self.config = config
        self.documents: list[Document] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: Optional[BM25Okapi] = None

        if chunks:
            self.build(chunks)

    def build(self, chunks: list[Any]) -> None:
        """
        Build BM25 index from chunks.
        """

        documents: list[Document] = []

        for chunk in chunks:
            document = ChunkAdapter.to_llama_document(chunk, self.config)
            if document is not None:
                documents.append(document)

        self.documents = documents
        self.tokenized_corpus = [
            TextTokenizer.tokenize(self._enriched_text(document))
            for document in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

    def search(self, query: str, k: int) -> list[SearchResult]:
        """
        Search BM25 index.
        """

        if self.bm25 is None:
            return []

        query_tokens = TextTokenizer.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked_indexes = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:k]
        results: list[SearchResult] = []

        for index in ranked_indexes:
            score = float(scores[index])
            if score <= 0:
                continue

            document = self.documents[index]
            payload = MetadataAdapter.normalize_payload(dict(document.metadata or {}), self.config)
            payload["retrieval_source"] = "bm25"

            results.append(
                SearchResultFactory.create(
                    id=f"bm25:{index}",
                    text=document.text,
                    score=score,
                    payload=payload,
                )
            )

        return results

    @staticmethod
    def _enriched_text(document: Document) -> str:
        """
        Build BM25 text enriched by metadata.
        """

        metadata = document.metadata or {}
        metadata_text = " ".join(
            str(value)
            for value in [
                metadata.get("article_number"),
                metadata.get("article_title"),
                metadata.get("header"),
                metadata.get("legal_domain"),
                " ".join(metadata.get("topics") or []),
                " ".join(metadata.get("keywords") or []),
            ]
            if value
        )
        return f"{metadata_text}\n{document.text}"


class LlamaIndexMetadataGraphRetriever(BaseGraphRetriever):
    """
    Metadata + context GraphRAG retriever.

    Graph expansion is triggered from high-confidence dense/BM25 seed articles.
    """

    def __init__(self, config: HybridRetrieverConfig, chunks: Optional[list[Any]] = None) -> None:
        self.config = config
        self.documents: list[Document] = []
        self.doc_id_to_docs: dict[str, list[int]] = {}
        self.doc_graph: dict[str, list[tuple[str, float, str | None]]] = {}
        self.topic_to_doc_ids: dict[str, set[str]] = {}
        self.keyword_to_doc_ids: dict[str, set[str]] = {}

        if chunks:
            self.build(chunks)

    def build(self, chunks: list[Any]) -> None:
        """
        Build graph from metadata and content references.
        """

        self.documents = []
        self.doc_id_to_docs = {}
        self.doc_graph = {}
        self.topic_to_doc_ids = {}
        self.keyword_to_doc_ids = {}

        for chunk in chunks:
            document = ChunkAdapter.to_llama_document(chunk, self.config)
            if document is None:
                continue

            index = len(self.documents)
            self.documents.append(document)
            metadata = document.metadata or {}
            doc_id = MetadataAdapter.get_document_id(metadata, self.config)

            if doc_id is None:
                continue

            self.doc_id_to_docs.setdefault(doc_id, []).append(index)
            self.doc_graph.setdefault(doc_id, [])

            self._index_topics_and_keywords(doc_id=doc_id, metadata=metadata)
            self._index_relations(doc_id=doc_id, metadata=metadata)

            if self.config.enable_related_article_edges:
                self._index_related_articles(doc_id=doc_id, metadata=metadata)

            if self.config.enable_tree_edges:
                self._index_tree_edges(doc_id=doc_id, metadata=metadata)

            if self.config.enable_content_edges:
                self._index_content_references(doc_id=doc_id, text=document.text)

        if self.config.enable_topic_edges:
            self._build_shared_topic_edges()
            self._build_shared_keyword_edges()

    def _index_topics_and_keywords(self, *, doc_id: str, metadata: dict[str, Any]) -> None:
        """
        Index topics and keywords for weak graph expansion.
        """

        topics = []
        keywords = []

        for key in self.config.topic_keys:
            topics.extend(MetadataAdapter.as_list(metadata.get(key)))

        for key in self.config.keyword_keys:
            keywords.extend(MetadataAdapter.as_list(metadata.get(key)))

        for key in self.config.nested_topic_keys:
            nested = metadata.get(key)
            if not isinstance(nested, dict):
                continue
            topics.extend(MetadataAdapter.as_list(nested.get("topics")))
            keywords.extend(MetadataAdapter.as_list(nested.get("keywords")))

        for topic in topics:
            topic_key = str(topic).lower().strip()
            if topic_key:
                self.topic_to_doc_ids.setdefault(topic_key, set()).add(doc_id)

        for keyword in keywords:
            keyword_key = str(keyword).lower().strip()
            if keyword_key:
                self.keyword_to_doc_ids.setdefault(keyword_key, set()).add(doc_id)

    def _index_relations(self, *, doc_id: str, metadata: dict[str, Any]) -> None:
        """
        Index graph_rag relations.
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
            if target_doc_id is None:
                continue

            relation_type = relation.get("type")
            self._add_edge(
                source_doc_id=doc_id,
                target_doc_id=target_doc_id,
                weight=self.config.relation_weight(relation_type),
                relation_type=str(relation_type) if relation_type else None,
            )

    def _index_related_articles(self, *, doc_id: str, metadata: dict[str, Any]) -> None:
        """
        Index explicit related_articles metadata.
        """

        for value in MetadataAdapter.as_list(metadata.get("related_articles")):
            target_doc_id = MetadataAdapter.normalize_id(value)
            if target_doc_id and target_doc_id != doc_id:
                self._add_edge(
                    source_doc_id=doc_id,
                    target_doc_id=target_doc_id,
                    weight=self.config.related_article_weight,
                    relation_type="RELATED_ARTICLE",
                )

    def _index_tree_edges(self, *, doc_id: str, metadata: dict[str, Any]) -> None:
        """
        Index tree_rag navigation/parent edges.
        """

        tree_rag = metadata.get("tree_rag")
        if not isinstance(tree_rag, dict):
            return

        navigation = tree_rag.get("navigation") or {}
        hierarchy = tree_rag.get("hierarchy") or {}

        for value in [navigation.get("next"), navigation.get("previous"), hierarchy.get("parent")]:
            target_doc_id = MetadataAdapter.normalize_id(value)
            if target_doc_id and target_doc_id != doc_id:
                self._add_edge(
                    source_doc_id=doc_id,
                    target_doc_id=target_doc_id,
                    weight=self.config.tree_edge_weight,
                    relation_type="TREE",
                )

    def _index_content_references(self, *, doc_id: str, text: str) -> None:
        """
        Index explicit references found in article text.
        """

        for target_doc_id in self._extract_article_references(text):
            if target_doc_id == doc_id:
                continue

            self._add_edge(
                source_doc_id=doc_id,
                target_doc_id=target_doc_id,
                weight=self.config.content_reference_weight,
                relation_type="CONTENT_REFERENCE",
            )

    def _build_shared_topic_edges(self) -> None:
        """
        Build weak graph edges between documents sharing topics.
        """

        self._build_shared_edges(self.topic_to_doc_ids, self.config.shared_topic_weight, "SHARED_TOPIC")

    def _build_shared_keyword_edges(self) -> None:
        """
        Build weak graph edges between documents sharing keywords.
        """

        self._build_shared_edges(self.keyword_to_doc_ids, self.config.shared_keyword_weight, "SHARED_KEYWORD")

    def _build_shared_edges(
        self,
        index: dict[str, set[str]],
        weight: float,
        relation_type: str,
    ) -> None:
        """
        Build weak edges from inverted metadata index.
        """

        for _, doc_ids in index.items():
            ordered_doc_ids = sorted(doc_ids)

            for source_doc_id in ordered_doc_ids:
                added = 0
                for target_doc_id in ordered_doc_ids:
                    if source_doc_id == target_doc_id:
                        continue

                    self._add_edge(
                        source_doc_id=source_doc_id,
                        target_doc_id=target_doc_id,
                        weight=weight,
                        relation_type=relation_type,
                    )
                    added += 1
                    if added >= self.config.max_topic_edges_per_doc:
                        break

    def _add_edge(
        self,
        *,
        source_doc_id: str,
        target_doc_id: str,
        weight: float,
        relation_type: str | None,
    ) -> None:
        """
        Add graph edge and keep the strongest duplicate edge.
        """

        if not source_doc_id or not target_doc_id:
            return

        edges = self.doc_graph.setdefault(source_doc_id, [])

        for index, (existing_target, existing_weight, _) in enumerate(edges):
            if existing_target != target_doc_id:
                continue
            if weight > existing_weight:
                edges[index] = (target_doc_id, weight, relation_type)
            return

        edges.append((target_doc_id, weight, relation_type))

    @staticmethod
    def _extract_article_references(text: str) -> list[str]:
        """
        Extract referenced article ids from legal text.
        """

        if not text:
            return []

        patterns = (
            r"\bст\.?\s*(\d+(?:\.\d+)?)",
            r"\bстать[ьяеию]+\s*(\d+(?:\.\d+)?)",
            r"\barticle[_\s-]*(\d+)(?:[_\.-](\d+))?",
        )

        references: list[str] = []
        seen: set[str] = set()

        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = f"{match.group(1)}.{match.group(2)}" if len(match.groups()) >= 2 and match.group(2) else match.group(1)
                value = value.strip()
                if value and value not in seen:
                    seen.add(value)
                    references.append(value)

        return references

    def _extract_relation_target(self, relation: dict[str, Any]) -> Optional[str]:
        """
        Extract relation target document id.
        """

        for key in self.config.relation_target_keys:
            normalized = MetadataAdapter.normalize_id(relation.get(key))
            if normalized is not None:
                return normalized
        return None

    def expand(self, seed_doc_ids: list[str], k: int) -> list[SearchResult]:
        """
        Expand from high-confidence seed articles through graph relations.
        """

        if not self.documents:
            return []

        results: list[SearchResult] = []
        seen_result_keys: set[str] = set()

        seed_doc_ids = [doc_id for doc_id in seed_doc_ids if doc_id in self.doc_graph][:self.config.graph_seed_top_n]

        for seed_rank, seed_doc_id in enumerate(seed_doc_ids, start=1):
            seed_base_score = 1.0 / seed_rank
            related = [
                edge for edge in sorted(self.doc_graph.get(seed_doc_id, []), key=lambda item: item[1], reverse=True)
                if edge[0] in self.doc_id_to_docs
            ][:self.config.graph_max_related_per_seed]

            for target_doc_id, relation_weight, relation_type in related:
                graph_score = seed_base_score * relation_weight * self.config.graph_relation_decay

                for doc_index in self.doc_id_to_docs.get(target_doc_id, []):
                    document = self.documents[doc_index]
                    key = f"{target_doc_id}:{doc_index}"

                    if key in seen_result_keys:
                        continue

                    seen_result_keys.add(key)
                    payload = MetadataAdapter.normalize_payload(dict(document.metadata or {}), self.config)
                    payload["retrieval_source"] = "graph"
                    payload["graph_seed_doc_id"] = seed_doc_id
                    payload["graph_relation_type"] = relation_type

                    results.append(
                        SearchResultFactory.create(
                            id=f"graph:{seed_doc_id}->{target_doc_id}:{doc_index}",
                            text=document.text,
                            score=graph_score,
                            payload=payload,
                        )
                    )

                    if len(results) >= k:
                        return results

        return results

    def search(self, query: str, k: int) -> list[SearchResult]:
        """
        Conservative backward-compatible graph search.
        """

        query_tokens = set(TextTokenizer.tokenize(query))
        doc_scores: dict[str, float] = {}

        for topic, doc_ids in self.topic_to_doc_ids.items():
            overlap = len(query_tokens & set(TextTokenizer.tokenize(topic)))
            if overlap <= 0:
                continue
            for doc_id in doc_ids:
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + overlap

        for keyword, doc_ids in self.keyword_to_doc_ids.items():
            overlap = len(query_tokens & set(TextTokenizer.tokenize(keyword)))
            if overlap <= 0:
                continue
            for doc_id in doc_ids:
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + overlap * 1.25

        ranked_doc_ids = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        return self.expand([doc_id for doc_id, _ in ranked_doc_ids], k=k)


class MetadataScoringService:
    """
    Scores candidate chunks using metadata and query context.
    """

    def __init__(self, config: HybridRetrieverConfig) -> None:
        self.config = config

    def score(self, query: str, result: SearchResult) -> float:
        """
        Return metadata relevance score in range 0..1.
        """

        payload = result.payload or {}
        query_tokens = set(TextTokenizer.tokenize(query))

        if not query_tokens:
            return 0.0

        header_score = self._field_overlap(query_tokens, [payload.get("header"), payload.get("article_title")])
        topic_score = self._field_overlap(query_tokens, payload.get("topics") or [])
        keyword_score = self._field_overlap(query_tokens, payload.get("keywords") or [])
        domain_score = self._field_overlap(query_tokens, [payload.get("legal_domain"), payload.get("chapter"), payload.get("paragraph")])
        related_score = 1.0 if self._query_mentions_related_article(query, payload) else 0.0

        score = (
            self.config.metadata_header_weight * header_score
            + self.config.metadata_topic_weight * topic_score
            + self.config.metadata_keyword_weight * keyword_score
            + self.config.metadata_domain_weight * domain_score
            + self.config.metadata_related_weight * related_score
        )

        return max(0.0, min(1.0, score))

    @staticmethod
    def _field_overlap(query_tokens: set[str], values: Any) -> float:
        """
        Compute token overlap between query and metadata values.
        """

        text = " ".join(MetadataAdapter.as_list(values))
        value_tokens = set(TextTokenizer.tokenize(text))

        if not value_tokens:
            return 0.0

        return len(query_tokens & value_tokens) / max(len(query_tokens), 1)

    @staticmethod
    def _query_mentions_related_article(query: str, payload: dict[str, Any]) -> bool:
        """
        Check whether query explicitly mentions this article or related article.
        """

        article = MetadataAdapter.normalize_id(payload.get("article_number"))
        related = {MetadataAdapter.normalize_id(item) for item in MetadataAdapter.as_list(payload.get("related_articles"))}
        related.discard(None)

        mentioned = set(LlamaIndexMetadataGraphRetriever._extract_article_references(query))

        return bool((article and article in mentioned) or (mentioned & related))


class AlphaFusionService:
    """
    Alpha fusion for dense, BM25, graph and metadata-aware scores.
    """

    def __init__(self, config: HybridRetrieverConfig) -> None:
        self.config = config
        self.metadata_scorer = MetadataScoringService(config)

    def fuse(
        self,
        *,
        query: str,
        dense_results: list[SearchResult],
        bm25_results: list[SearchResult],
        graph_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Fuse retrieval results.
        """

        dense_results = self._rank_normalize(dense_results)
        bm25_results = self._rank_normalize(bm25_results)
        graph_results = self._rank_normalize(graph_results)

        merged: dict[str, SearchResult] = {}
        scores: dict[str, float] = {}

        self._add_results(merged=merged, scores=scores, results=dense_results, weight=self.config.alpha, source="dense")
        self._add_results(merged=merged, scores=scores, results=bm25_results, weight=self.config.bm25_weight, source="bm25")
        self._add_results(merged=merged, scores=scores, results=graph_results, weight=self.config.graph_weight, source="graph")

        fused = list(merged.values())

        for result in fused:
            key = self._dedupe_key(result)
            metadata_score = self.metadata_scorer.score(query, result)
            result.payload = result.payload or {}
            result.payload["metadata_score"] = round(metadata_score, 4)
            result.score = scores.get(key, 0.0) + self.config.metadata_weight * metadata_score

        filtered = self._basic_filter(fused)
        filtered.sort(key=lambda item: item.score, reverse=True)

        diversified = self._diversify_by_article(
            hits=filtered,
            top_k=top_k,
        )

        return diversified[:top_k]

    @staticmethod
    def _rank_normalize(results: list[SearchResult]) -> list[SearchResult]:
        """
        Convert ranking position to stable 0..1 score.
        """

        for rank, result in enumerate(results or [], start=1):
            result.score = 1.0 / rank
        return results

    def _add_results(
        self,
        *,
        merged: dict[str, SearchResult],
        scores: dict[str, float],
        results: list[SearchResult],
        weight: float,
        source: str,
    ) -> None:
        """
        Add weighted source results.
        """

        for result in results:
            key = self._dedupe_key(result)
            weighted_score = float(result.score or 0.0) * weight

            if key not in merged:
                merged[key] = result
                scores[key] = weighted_score
                payload = merged[key].payload or {}
                payload["retrieval_sources"] = [source]
                merged[key].payload = payload
                continue

            scores[key] += weighted_score
            merged[key].payload = self._merge_payloads(merged[key].payload or {}, result.payload or {}, source)

    def _basic_filter(self, hits: list[SearchResult]) -> list[SearchResult]:
        """
        Remove empty, short and duplicate results.
        """

        seen: set[str] = set()
        result: list[SearchResult] = []

        for hit in hits:
            text = (hit.text or "").strip()
            if len(text) < self.config.min_text_len:
                continue

            key = self._dedupe_key(hit)
            if key in seen:
                continue

            seen.add(key)
            result.append(hit)

        return result

    def _diversify_by_article(
        self,
        *,
        hits: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Limit repeated chunks from the same article after fusion.

        Why:
        - without this step top-5 may contain 3-4 chunks from one article;
        - article-level coverage becomes worse;
        - reranker receives less diverse legal context.

        Strategy:
        1. First pass keeps no more than max_chunks_per_article per article.
        2. Second pass fills remaining slots if the first pass was too strict.
        """

        selected: list[SearchResult] = []
        overflow: list[SearchResult] = []
        article_counts: dict[str, int] = {}

        for hit in hits:
            payload = hit.payload or {}
            article = str(
                payload.get("retrieval_doc_id")
                or payload.get("article_number")
                or "unknown"
            )

            current_count = article_counts.get(article, 0)

            if current_count < self.config.max_chunks_per_article:
                selected.append(hit)
                article_counts[article] = current_count + 1
            else:
                overflow.append(hit)

            if len(selected) >= top_k:
                return selected

        for hit in overflow:
            selected.append(hit)
            if len(selected) >= top_k:
                break

        return selected

    @staticmethod
    def _dedupe_key(result: SearchResult) -> str:
        """
        Build stable dedupe key.
        """

        payload = result.payload or {}
        doc_id = payload.get("retrieval_doc_id") or payload.get("article_number")
        header = payload.get("header")

        if doc_id and header:
            return f"doc:{doc_id}|header:{header}"

        if getattr(result, "id", None):
            return str(result.id)

        text = (getattr(result, "text", "") or "")[:500]
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _merge_payloads(left: dict[str, Any], right: dict[str, Any], source: str) -> dict[str, Any]:
        """
        Merge payloads from duplicate results.
        """

        merged = dict(left)

        for key, value in right.items():
            if key not in merged:
                merged[key] = value

        sources = set(merged.get("retrieval_sources", []))
        sources.add(source)

        existing_source = merged.get("retrieval_source")
        if existing_source:
            sources.add(existing_source)

        merged["retrieval_sources"] = sorted(sources)
        return merged


class Retriever(BaseRetriever):
    """
    Hybrid Retriever:
    - Dense Qdrant
    - BM25 sparse
    - metadata/context graph expansion
    - metadata-aware AlphaFusion
    """

    def __init__(
        self,
        vector_store: Any,
        embedder: Embedder,
        *,
        chunks: Optional[list[Any]] = None,
        config: Optional[HybridRetrieverConfig] = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.config = config or HybridRetrieverConfig()

        self.dense = QdrantDenseRetriever(vector_store=vector_store, config=self.config)
        self.bm25 = BM25SparseRetriever(config=self.config, chunks=chunks)
        self.graph = LlamaIndexMetadataGraphRetriever(config=self.config, chunks=chunks)
        self.fusion = AlphaFusionService(self.config)

    def build_sparse_and_graph(self, chunks: list[Any]) -> None:
        """
        Build BM25 and graph indexes after ingestion.
        """

        self.bm25.build(chunks)
        self.graph.build(chunks)

    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        Retrieve using Dense + BM25 + graph expansion + metadata scoring.
        """

        query = (query or "").strip()
        if not query:
            return []

        pool_size = min(self.config.max_pool_size, max(top_k * self.config.pool_multiplier, 30))
        query_vec = self.embedder.encode_queries([query])[0]

        dense_candidates = self.dense.search(query_vec=query_vec, k=pool_size)
        bm25_candidates = self.bm25.search(query=query, k=pool_size)

        seed_doc_ids = self._select_graph_seed_doc_ids(
            dense_candidates=dense_candidates,
            bm25_candidates=bm25_candidates,
        )

        graph_candidates = self.graph.expand(seed_doc_ids=seed_doc_ids, k=pool_size)

        return self.fusion.fuse(
            query=query,
            dense_results=dense_candidates,
            bm25_results=bm25_candidates,
            graph_results=graph_candidates,
            top_k=top_k,
        )

    def _select_graph_seed_doc_ids(
        self,
        *,
        dense_candidates: list[SearchResult],
        bm25_candidates: list[SearchResult],
    ) -> list[str]:
        """
        Select graph expansion seeds from high-confidence dense/BM25 candidates.
        """

        candidates = dense_candidates[:self.config.graph_seed_top_n]
        candidates += bm25_candidates[:max(2, self.config.graph_seed_top_n // 2)]

        seed_doc_ids: list[str] = []
        seen: set[str] = set()

        for candidate in candidates:
            payload = candidate.payload or {}
            doc_id = payload.get("retrieval_doc_id") or payload.get("article_number")
            doc_id = MetadataAdapter.normalize_id(doc_id)

            if doc_id is None or doc_id in seen:
                continue

            seen.add(doc_id)
            seed_doc_ids.append(doc_id)

        return seed_doc_ids

    def debug_query(self, query: str, top_k: int = 10) -> None:
        """
        Print hybrid debug output.
        """

        print("\n" + "=" * 100)
        print("[HYBRID RETRIEVAL DEBUG: DENSE + BM25 + GRAPH + METADATA]")
        print(f"QUERY: {query}")
        print("=" * 100)

        hits = self.retrieve(query=query, top_k=top_k)

        if not hits:
            print("No hits")
            return

        for index, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            print("\n" + "-" * 100)
            print(f"TOP {index}")
            print(f"SCORE   : {hit.score:.4f}")
            print(f"SOURCES : {payload.get('retrieval_sources')}")
            print(f"META    : {payload.get('metadata_score')}")
            print(f"DOC ID  : {payload.get('retrieval_doc_id')}")
            print(f"ARTICLE : {payload.get('article_number', 'unknown')}")
            print(f"HEADER  : {payload.get('header', 'unknown')}")
            print(f"TOPICS  : {payload.get('topics')}")
            print(f"KEYWORDS: {payload.get('keywords')}")
            print(f"GRAPH SEED: {payload.get('graph_seed_doc_id')}")
            print(f"GRAPH REL : {payload.get('graph_relation_type')}")
            print(f"ID      : {hit.id}")
            print("\nTEXT:\n")
            print((hit.text or "")[:1200])
