from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Tuple

import math
import re

from sentence_transformers import CrossEncoder
from rag.search_result import SearchResult


class BaseReranker(ABC):

    @abstractmethod
    def rerank(
        self,
        query: str,
        hits: List["SearchResult"],
        *,
        top_n: int
    ) -> List["SearchResult"]:
        raise NotImplementedError


class Reranker(BaseReranker):

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        batch_size: int = 8,
        max_length: int = 512,
        top_n: int = 5,

        rerank_weight: float = 0.60,
        dense_weight: float = 0.30,
        lexical_weight: float = 0.10,

        exact_header_boost: float = 0.30,
        partial_header_boost: float = 0.15,
        generic_header_penalty: float = 0.03,
        low_lexical_penalty: float = 0.02,

        max_chunks_per_article: int = 2,
    ):

        self.model = self._load(model_name, max_length)

        self.batch_size = batch_size
        self.max_length = max_length
        self.top_n = top_n

        self.rerank_weight = rerank_weight
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

        self.exact_header_boost = exact_header_boost
        self.partial_header_boost = partial_header_boost

        self.generic_header_penalty = generic_header_penalty
        self.low_lexical_penalty = low_lexical_penalty

        self.max_chunks_per_article = max_chunks_per_article


    @staticmethod
    @lru_cache(maxsize=1)
    def _load(model_name: str, max_length: int) -> CrossEncoder:
        model = CrossEncoder(model_name, max_length=max_length)

        tokenizer = model.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.model.config.pad_token_id = tokenizer.pad_token_id
        return model

    def rerank(
        self,
        query: str,
        hits: List[SearchResult],
        top_n: int | None = None
    ) -> List[SearchResult]:

        if not hits:
            return []

        query = (query or "").strip()
        if not query:
            return []

        top_n = top_n or self.top_n

        valid_hits = [h for h in hits if h.text and h.text.strip()]
        if not valid_hits:
            return []

        pairs = [self._build_pair(query, h) for h in valid_hits]

        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        scored: List[SearchResult] = []

        for hit, raw in zip(valid_hits, raw_scores):

            rerank_score = self._normalize_logit(raw)
            dense_score = self._normalize_dense(getattr(hit, "score", 0.0))
            lexical_score = self._lexical_score(query, hit.text or "")

            header = (hit.payload or {}).get("header", "")

            header_score = self._header_score(query, header)
            penalty = self._penalty_score(query, header, hit.text or "")

            final_score = (
                self.rerank_weight * rerank_score +
                self.dense_weight * dense_score +
                self.lexical_weight * lexical_score +
                header_score -
                penalty
            )

            scored.append(
                SearchResult.from_rerank(hit, final_score)
            )

        scored.sort(key=lambda x: x.score, reverse=True)

        # ❗ ВАЖНО: только diversity, без threshold filter
        diversified = self._diversify(scored, top_n)

        return diversified[:top_n]

    # =========================
    # PAIR BUILD
    # =========================

    def _build_pair(self, query: str, doc: SearchResult) -> Tuple[str, str]:

        p = doc.payload or {}

        article = p.get("article_number", "")
        header = p.get("header", "")
        text = self._prepare_text(doc.text)

        doc_text = f"""
        Статья: {article}
        Заголовок: {header}
        Текст: {text}
        """.strip()

        return query, doc_text

    def _prepare_text(self, text: str) -> str:
        text = (text or "").strip()
        text = re.sub(r"\s+", " ", text)

        if len(text) <= 2000:
            return text

        return text[:1400] + "\n...\n" + text[-400:]


    def _normalize_logit(self, x: float) -> float:
        return float(math.tanh(float(x) / 2.0))

    def _normalize_dense(self, x: float) -> float:
        return max(0.0, min(1.0, float(x)))


    def _tokenize(self, text: str) -> List[str]:
        return [
            w for w in re.findall(r"\w+", text.lower())
            if len(w) > 2
        ]

    def _lexical_score(self, query: str, text: str) -> float:
        q = set(self._tokenize(query))
        t = set(self._tokenize(text[:800]))

        if not q:
            return 0.0

        return len(q & t) / len(q)


    def _header_score(self, query: str, header: str) -> float:

        q = query.lower().strip()
        h = (header or "").lower().strip()

        if not q or not h:
            return 0.0

        if q == h:
            return self.exact_header_boost

        if q in h:
            return self.partial_header_boost

        q_tokens = set(self._tokenize(q))
        h_tokens = set(self._tokenize(h))

        if not q_tokens:
            return 0.0

        return (len(q_tokens & h_tokens) / len(q_tokens)) * 0.15


    def _penalty_score(self, query: str, header: str, text: str) -> float:

        penalty = 0.0

        generic = {
            "общие положения",
            "понятие",
            "краткое содержание",
        }

        if (header or "").lower().strip() in generic:
            penalty += self.generic_header_penalty

        if self._lexical_score(query, text) < 0.1:
            penalty += self.low_lexical_penalty

        return penalty


    def _diversify(
        self,
        hits: List[SearchResult],
        top_n: int
    ) -> List[SearchResult]:

        selected = []
        counts = {}

        for h in hits:

            article = (h.payload or {}).get("article_number", "unknown")

            if counts.get(article, 0) >= self.max_chunks_per_article:
                continue

            selected.append(h)
            counts[article] = counts.get(article, 0) + 1

            if len(selected) >= top_n:
                break

        return selected


    def debug_rerank(self, query: str, hits: List[SearchResult], top_n: int = 10):
        print("\n" + "=" * 80)
        print(f"RERANK DEBUG: {query}")
        print("=" * 80)

        ranked = self.rerank(query, hits, top_n=top_n)

        for i, h in enumerate(ranked, 1):
            p = h.payload or {}
            print(f"\n[{i}] score={h.score:.4f} article={p.get('article_number')}")
            print(p.get("header"))