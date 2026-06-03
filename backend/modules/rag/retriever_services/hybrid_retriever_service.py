import hashlib
import math
from dataclasses import dataclass
from typing import Any

from backend.modules.rag.search_result_service import SearchResult


@dataclass(frozen=True, slots=True)
class HybridRetrieverConfig:
    """
    Configuration for Dense + BM25 + metadata GraphRAG hybrid retrieval.

    The config is intentionally domain-neutral, so the retriever can work
    not only with labor law, but also with civil, criminal, administrative,
    tax or other legal corpora.
    """

    alpha: float = 0.8
    graph_weight: float = 0.15
    pool_multiplier: int = 8
    max_pool_size: int = 80
    min_text_len: int = 40

    document_id_keys: tuple[str, ...] = (
        "article_number",
        "article",
        "article_id",
        "norm_id",
        "document_id",
        "section_number",
        "paragraph_number",
    )

    header_keys: tuple[str, ...] = (
        "header",
        "title",
        "heading",
        "name",
    )

    topic_keys: tuple[str, ...] = (
        "topics",
    )

    nested_topic_keys: tuple[str, ...] = (
        "classic_rag",
    )

    graph_metadata_key: str = "graph_rag"

    relation_target_keys: tuple[str, ...] = (
        "target",
        "to",
        "article",
        "article_number",
        "document_id",
        "norm_id",
    )

class AlphaFusionService:
    """
    Alpha fusion for dense, BM25 and graph results.

    Formula:
        final_score =
            alpha * dense_score
            + (1 - alpha) * bm25_score
            + graph_weight * graph_score
    """

    def __init__(
            self,
            config: HybridRetrieverConfig
    ) -> None:
        """
        Initialize fusion service.
        """

        self.config = config

    def fuse(
            self,
            *,
            dense_results: list[SearchResult],
            bm25_results: list[SearchResult],
            graph_results: list[SearchResult],
            top_k: int
    ) -> list[SearchResult]:
        """
        Fuse retrieval results.
        """

        dense_results = self._normalize_scores(dense_results)
        bm25_results = self._normalize_scores(bm25_results)
        graph_results = self._normalize_scores(graph_results)

        merged: dict[str, SearchResult] = {}
        scores: dict[str, float] = {}

        self._add_results(
            merged=merged,
            scores=scores,
            results=dense_results,
            weight=self.config.alpha,
            source="dense"
        )

        self._add_results(
            merged=merged,
            scores=scores,
            results=bm25_results,
            weight=1.0 - self.config.alpha,
            source="bm25"
        )

        self._add_results(
            merged=merged,
            scores=scores,
            results=graph_results,
            weight=self.config.graph_weight,
            source="graph"
        )

        fused = list(merged.values())

        for result in fused:
            key = self._dedupe_key(result)
            result.score = scores.get(key, 0.0)

        filtered = self._basic_filter(fused)

        filtered.sort(
            key=lambda item: item.score,
            reverse=True
        )

        return filtered[:top_k]

    def _add_results(
            self,
            *,
            merged: dict[str, SearchResult],
            scores: dict[str, float],
            results: list[SearchResult],
            weight: float,
            source: str
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

            else:
                scores[key] += weighted_score

                merged[key].payload = self._merge_payloads(
                    merged[key].payload or {},
                    result.payload or {},
                    source
                )

    def _normalize_scores(
            self,
            results: list[SearchResult]
    ) -> list[SearchResult]:
        """
        Normalize scores to 0..1.
        """

        if not results:
            return []

        raw_scores = [
            float(result.score or 0.0)
            for result in results
        ]

        min_score = min(raw_scores)
        max_score = max(raw_scores)

        if math.isclose(max_score, min_score):
            for result in results:
                result.score = 1.0

            return results

        for result in results:
            result.score = (
                float(result.score or 0.0)
                - min_score
            ) / (
                max_score
                - min_score
            )

        return results

    def _basic_filter(
            self,
            hits: list[SearchResult]
    ) -> list[SearchResult]:
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

    @staticmethod
    def _dedupe_key(
            result: SearchResult
    ) -> str:
        """
        Build stable dedupe key.
        """

        payload = result.payload or {}

        doc_id = payload.get("retrieval_doc_id")
        header = payload.get("header")

        if doc_id and header:
            return f"doc:{doc_id}|header:{header}"

        if getattr(result, "id", None):
            return str(result.id)

        text = (getattr(result, "text", "") or "")[:500]

        return hashlib.md5(
            text.encode()
        ).hexdigest()

    @staticmethod
    def _merge_payloads(
            left: dict[str, Any],
            right: dict[str, Any],
            source: str
    ) -> dict[str, Any]:
        """
        Merge payloads.
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