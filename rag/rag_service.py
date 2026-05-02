from typing import List

from rag.rag_config import RAGResponse
from rag.search_result import SearchResult


class RAGService:

    def __init__(self, retriever, reranker, generator):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    def ask(self, query: str) -> RAGResponse:

        hits = self.retriever.retrieve(query, top_k=25)

        hits = self._deduplicate(hits)

        print("\n--- RETRIEVER ---")
        for h in hits[:10]:
            print(f"[{h.score:.4f}] {h.text[:80]}...")

        reranked = self.reranker.rerank(
            query=query,
            hits=hits,
            top_n=6
        )

        print("\n--- RERANKED ---")
        for h in reranked:
            print(f"[{h.score:.4f}] {h.text[:80]}...")

        context = self._build_context(reranked)

        answer = self.generator.generate(query, context)

        return RAGResponse(
            answer=answer,
            sources=[h.payload for h in reranked]
        )

    def _build_context(self, hits: List[SearchResult]) -> str:

        grouped = {}

        for h in hits:
            article = h.payload.get("article_number") or "unknown"
            grouped.setdefault(article, []).append(h)

        parts = []

        for article, items in grouped.items():

            block = [f"СТАТЬЯ {article}"]

            for h in items:
                header = h.payload.get("header") or ""
                text = (h.text or "").strip()

                if len(text) < 30:
                    continue

                block.append(f"### {header}\n{text}")

            parts.append("\n\n".join(block))

        return "\n\n---\n\n".join(parts)

    def _deduplicate(self, hits: List[SearchResult]) -> List[SearchResult]:

        seen = set()
        result = []

        for h in hits:
            text = (h.text or "").strip()

            if len(text) < 30:
                continue

            key = hash(text[:200])

            if key in seen:
                continue

            seen.add(key)
            result.append(h)

        return result
