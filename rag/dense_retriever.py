import re
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import lru_cache
from typing import List, Set
import hashlib

from llama_index.core import VectorStoreIndex
from llama_index.core.indices.vector_store import VectorIndexRetriever
from sentence_transformers import SentenceTransformer

from rag.search_result import SearchResult


class BaseDenseRetriever(ABC):
    """
    Abstract interface for dense vector retrievers.

    Dense retrievers perform semantic similarity search
    using embedding vectors.
    """

    @abstractmethod
    def search(
        self,
        query_vec: list[float],
        k: int
    ) -> list[SearchResult]:
        """
        Execute dense vector similarity search.

        Args:
            query_vec (List[float]):
                Query embedding vector.

            k (int):
                Number of documents to retrieve.

        Returns:
            List[SearchResult]:
                Retrieved search results.
        """

        raise NotImplementedError


class BaseRetriever(ABC):
    """
    Abstract interface for retrievers.

    Retriever converts text query into embeddings
    and returns relevant search results.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query (str):
                User query.

            top_k (int):
                Number of chunks to return.

        Returns:
            List[SearchResult]:
                Retrieved chunks.
        """

        raise NotImplementedError


class Embedder:
    """
    Wrapper around SentenceTransformer models.

    Responsible for:
    - loading embedding model
    - query/document encoding
    - E5 prefix handling
    - vector normalization
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        normalize: bool = True
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        self._model = self._load_model(model_name)

        self.dim = (
            self._model.get_sentence_embedding_dimension()
        )

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_model(model_name: str) -> SentenceTransformer:
        """
        Load and cache embedding model.

        Args:
            model_name (str):
                HuggingFace model name.

        Returns:
            SentenceTransformer:
                Loaded embedding model.
        """

        return SentenceTransformer(model_name)

    def encode_queries(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """
        Encode search queries.

        Args:
            texts (List[str]):
                Query texts.

        Returns:
            List[List[float]]:
                Query embeddings.
        """


        texts = self._apply_prefix(
            texts,
            is_query=True
        )

        return self._encode(texts)

    def encode_passages(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """
        Encode document passages.

        Args:
            texts (List[str]):
                Passage texts.

        Returns:
            List[List[float]]:
                Passage embeddings.
        """

        texts = self._apply_prefix(
            texts,
            is_query=False
        )

        return self._encode(texts)

    def _apply_prefix(
        self,
        texts: List[str],
        is_query: bool
    ) -> List[str]:
        """
        Apply E5 prefixes if model requires them.

        E5 models require:
        - "query: " for queries
        - "passage: " for documents

        Args:
            texts (List[str]):
                Input texts.

            is_query (bool):
                Whether texts are queries.

         Returns:
            List[str]:
                Prefixed texts.
        """

        if "e5" not in self.model_name.lower():
            return texts

        prefix = (
            "query: "
            if is_query
            else "passage: "
        )

        return [
            prefix + t
            for t in texts
        ]

    def _encode(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """
        Encode texts into embeddings.

        Args:
            texts (List[str]):
                Input texts.

        Returns:
            List[List[float]]:
                Embedding vectors.
        """

        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
        )

        return vectors.tolist()


class QdrantDenseRetriever(BaseDenseRetriever):
    """
    Dense semantic retriever over Qdrant.

    Performs cosine similarity search
    using embedding vectors.
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def search(
        self,
        query_vec: list[float],
        k: int
    ) -> list[SearchResult]:
        """
        Execute vector similarity search.

        Args:
            query_vec (List[float]):
                Query embedding vector.

            k (int):
                Number of documents to retrieve.

        Returns:
            List[SearchResult]:
                Retrieved search results.
        """

        hits = self.vector_store.search(
            query_vector=query_vec,
            limit=k
        )

        results = []

        for hit in hits:

            sr = SearchResult.from_qdrant(hit)

            if sr.text and sr.text.strip():
                results.append(sr)

        return results

class LightRAGEnhancer:
    """
    Semantic enhancement layer for retrieval (LlamaIndex-based version).

    This component preserves original logic:
    - semantic query expansion
    - legal synonym expansion
    - neighbor-based chunk expansion
    - article graph construction
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        neighbor_window: int = 1,
        max_expansion: int = 20
    ) -> None:
        """
        Args:
            index (VectorStoreIndex):
                LlamaIndex vector index (Qdrant / in-memory / etc.)

            neighbor_window (int):
                Number of neighboring articles/chunks to include.

            max_expansion (int):
                Max number of expanded candidates.
        """

        self.index = index
        self.neighbor_window = neighbor_window
        self.max_expansion = max_expansion

        self.legal_synonyms = {
            "должник": [
                "обязанное лицо",
                "исполнитель обязательства"
            ],
            "кредитор": [
                "управомоченное лицо",
                "получатель исполнения"
            ],
            "обязательство": [
                "гражданское обязательство",
                "договорное обязательство"
            ],
            "договор": [
                "соглашение",
                "контракт",
            ],
            "убытки": [
                "вред",
                "ущерб",
                "компенсация"
            ],
            "неустойка": [
                "штраф",
                "пеня"
            ],
        }

    def enrich_query(self, query: str) -> str:
        """
        Expand query using legal synonyms and article normalization.
        """

        query = query.strip().lower()

        expanded_terms: Set[str] = set()
        expanded_terms.add(query)

        words = re.findall(r"\w+", query)

        for word in words:
            if word in self.legal_synonyms:
                expanded_terms.update(self.legal_synonyms[word])

        article_match = re.findall(
            r"(?:статья|ст\.?)\s*(\d+(?:\.\d+)?)",
            query,
            flags=re.IGNORECASE
        )

        for art in article_match:
            expanded_terms.add(f"статья {art}")
            expanded_terms.add(f"гк рф статья {art}")
            expanded_terms.add(f"норма {art}")

        return " ".join(expanded_terms)

    def expand_candidates(
        self,
        candidates: List[SearchResult]
    ) -> List[SearchResult]:

        if not candidates:
            return []

        expanded: List[SearchResult] = []
        seen_ids: Set[str] = set()

        article_groups: dict[str, List[SearchResult]] = defaultdict(list)

        for cand in candidates:

            if not cand:
                continue

            expanded.append(cand)

            if cand.id:
                seen_ids.add(cand.id)

            article = getattr(cand, "article_number", None)

            if article:
                article_groups[article].append(cand)

        article_numbers = []

        for art in article_groups.keys():
            try:
                article_numbers.append(float(art))
            except Exception:
                continue

        article_numbers = sorted(article_numbers)

        article_lookup = {
            float(k): v
            for k, v in article_groups.items()
            if self._is_float(k)
        }

        for art_num in article_numbers:

            for offset in range(
                -self.neighbor_window,
                self.neighbor_window + 1
            ):
                if offset == 0:
                    continue

                neighbor = art_num + offset

                if neighbor not in article_lookup:
                    continue

                for candidate in article_lookup[neighbor]:

                    if candidate.id and candidate.id in seen_ids:
                        continue

                    expanded.append(candidate)

                    if candidate.id:
                        seen_ids.add(candidate.id)

        expanded = self._deduplicate(expanded)

        return expanded[:self.max_expansion]

    def build_graph(
        self,
        candidates: list[SearchResult]
    ) -> dict[str, set[str]]:

        graph: dict[str, set[str]] = defaultdict(set)

        articles = []

        for c in candidates:
            article = getattr(c, "article_number", None)
            if article:
                articles.append(article)

        unique_articles = sorted(
            set(articles),
            key=lambda x: float(x) if self._is_float(x) else 0
        )

        for idx, article in enumerate(unique_articles):

            if idx > 0:
                graph[article].add(unique_articles[idx - 1])

            if idx < len(unique_articles) - 1:
                graph[article].add(unique_articles[idx + 1])

        return graph


    def _deduplicate(
        self,
        candidates: List[SearchResult]
    ) -> List[SearchResult]:

        seen: Set[str] = set()
        result: List[SearchResult] = []

        for c in candidates:

            key = c.id or (c.text[:150] if c.text else "")

            if key in seen:
                continue

            seen.add(key)
            result.append(c)

        return result


    @staticmethod
    def _is_float(value: str) -> bool:
        try:
            float(value)
            return True
        except Exception:
            return False


class Retriever(BaseRetriever):
    """
    Main semantic retriever (LlamaIndex-based).

    Pipeline:
    1. Query enrichment (optional)
    2. LlamaIndex retrieval (auto embeddings)
    3. Filtering
    4. Optional graph expansion
    """

    def __init__(
        self,
        vector_store,
        *,
        max_pool_size: int = 80,
        min_text_len: int = 40,
        use_enhancer: bool = True
    ):

        self.vector_store = vector_store

        self.max_pool_size = max_pool_size
        self.min_text_len = min_text_len

        self.index = VectorStoreIndex.from_vector_store(vector_store)

        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=max_pool_size
        )

        self.enhancer = None
        if use_enhancer:
            self.enhancer = LightRAGEnhancer(index=self.index)

    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[SearchResult]:

        query = (query or "").strip()
        if not query:
            return []

        if self.enhancer:
            query = self.enhancer.enrich_query(query)

        nodes = self.retriever.retrieve(query)

        candidates: List[SearchResult] = []

        for n in nodes:
            node = n.node

            text = node.get_content()

            candidates.append(
                SearchResult(
                    id=getattr(node, "id_", None),
                    text=text,
                    score=float(getattr(n, "score", 0.0) or 0.0),
                    payload={}
                )
            )

        candidates = self._basic_filter(candidates)

        if self.enhancer:
            candidates = self.enhancer.expand_candidates(candidates)

        return candidates[:top_k]

    def _basic_filter(
        self,
        hits: List[SearchResult]
    ) -> List[SearchResult]:

        seen: Set[str] = set()
        result: List[SearchResult] = []

        for h in hits:

            text = (h.text or "").strip()
            if len(text) < self.min_text_len:
                continue

            key = h.id or hashlib.md5(text[:200].encode()).hexdigest()

            if key in seen:
                continue

            seen.add(key)
            result.append(h)

        return result

    def debug_query(
        self,
        query: str,
        top_k: int = 10
    ) -> None:

        print("\n" + "=" * 80)
        print(f"[QUERY] {query}")

        if not query:
            print("Empty query")
            return

        nodes = self.retriever.retrieve(query)

        print(f"\n[LlamaIndex TOP {top_k}]")

        for i, n in enumerate(nodes[:top_k], start=1):

            print(
                f"{i}. score={getattr(n, 'score', 0.0):.4f} "
                f"| id={getattr(n.node, 'id_', None)}"
            )

            print(n.node.get_content()[:400])
            print()