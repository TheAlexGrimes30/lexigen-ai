import re
from enum import Enum

from backend.modules.rag.generator import ContractRiskAnalysisPromptBuilder, CreditPromptBuilder
from backend.modules.rag.rag_config import RAGResponse
from backend.modules.rag.search_result_service import SearchResult

class RAGMode(str, Enum):
    USER_QUERY = "user_query"
    DOCUMENT_ANALYSIS = "document_analysis"

class RAGService:
    """Сервис RAG с разными параметрами для вопроса и анализа документа."""

    def __init__(
        self,
        retriever,
        reranker,
        generator,
        max_context_chars: int = 1200,
        min_final_score: float = 0.50,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.max_context_chars = max_context_chars
        self.min_final_score = min_final_score

    def ask(
        self,
        query: str,
        mode: RAGMode = RAGMode.USER_QUERY,
    ) -> RAGResponse:
        settings = self._settings_for_mode(mode)

        hits = self.retriever.retrieve(
            query=query,
            top_k=settings["top_k"],
        )

        reranked = (
            self.reranker.rerank(
                query=query,
                hits=hits,
                top_n=settings["top_n"],
            )
            if settings["use_reranker"]
            else hits[:settings["top_n"]]
        )

        filtered = self._filter_hits(reranked)

        if not filtered:
            filtered = reranked[:settings["top_n"]]

        context = self._build_context(
            hits=filtered,
            max_context_chars=settings["max_context_chars"],
        )

        if len(context.strip()) < 80:
            context = self._fallback_context(filtered)

        context = self._sanitize_context(context)

        old_prompt_builder = self.generator.prompt_builder
        self.generator.prompt_builder = settings["prompt_builder"]

        try:
            raw_answer = self.generator.generate(
                query=query,
                context=context,
                hits=filtered,
                max_tokens=settings["max_tokens"],
            )
        finally:
            self.generator.prompt_builder = old_prompt_builder

        answer = self._validate_and_fix(raw_answer, filtered)

        sources = self._build_sources(filtered)

        if sources:
            answer = f"{answer}\n\nИсточник: {sources[0]}."

        return RAGResponse(
            answer=answer,
            sources=sources,
        )

    def _settings_for_mode(self, mode: RAGMode) -> dict:
        if mode == RAGMode.USER_QUERY:
            return {
                "top_k": 25,
                "top_n": 10,
                "max_context_chars": 3500,
                "use_reranker": False,
                "max_tokens": 512,
                "prompt_builder": CreditPromptBuilder(),
            }

        return {
            "top_k": 8,
            "top_n": 4,
            "max_context_chars": 1200,
            "use_reranker": False,
            "max_tokens": 400,
            "prompt_builder": ContractRiskAnalysisPromptBuilder(),
        }

    def _sanitize_context(self, text: str) -> str:
        text = re.sub(r"(?i)\b(a:|q:)\b", "", text)
        text = re.sub(r"\bНедостаточно данных\b.*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    def _filter_hits(self, hits: list[SearchResult]) -> list[SearchResult]:
        filtered = []
        seen: set[tuple] = set()

        for hit in hits:
            article = hit.payload.get("article_number")

            if not article:
                continue

            score = getattr(hit, "final_score", hit.score)

            if score < self.min_final_score:
                continue

            header = (hit.payload.get("header") or "").lower()
            key = (article, header)

            if key in seen:
                continue

            seen.add(key)
            filtered.append(hit)

            if len(filtered) >= 6:
                break

        return filtered

    def _build_context(
        self,
        hits: list[SearchResult],
        max_context_chars: int,
    ) -> str:
        parts = []
        size = 0
        seen = set()

        for hit in hits:
            text = (hit.text or "").strip()

            if len(text) < 40:
                continue

            norm = self._normalize(text)

            if norm in seen:
                continue

            seen.add(norm)

            article = hit.payload.get("article_number", "?")
            header = hit.payload.get("header", "")
            source = hit.payload.get("source", "Нормативный акт")

            block = f"""[СТАТЬЯ {article} — {source}]
            {header}
            
            {text[:900]}""".strip()

            if size + len(block) > max_context_chars:
                break

            parts.append(block)
            size += len(block)

        return "\n\n".join(parts)

    def _fallback_context(self, hits: list[SearchResult]) -> str:
        parts = []

        for hit in hits:
            text = (hit.text or "").strip()

            if len(text) < 60:
                continue

            article = hit.payload.get("article_number", "?")
            source = hit.payload.get("source", "Нормативный акт")

            parts.append(
                f"[СТАТЬЯ {article} — {source}]\n{text[:500]}"
            )

        return "\n\n".join(parts)

    def _validate_and_fix(self, text: str, hits: list[SearchResult]) -> str:
        if not text:
            return "Недостаточно данных."

        text = text.strip()
        text = re.sub(r"(?i)^(a:|q:|ответ:)\s*", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        allowed_articles = {
            str(hit.payload.get("article_number"))
            for hit in hits
            if hit.payload.get("article_number")
        }

        def fix_article(match):
            article = match.group(1)
            return f"статья {article}" if article in allowed_articles else "статья ?"

        text = re.sub(r"статья\s+(\d+)", fix_article, text, flags=re.IGNORECASE)

        if len(text.split()) < 3:
            return "Недостаточно данных."

        return text.strip()

    def _build_sources(self, hits: list[SearchResult]) -> list[str]:
        sources = []
        seen = set()

        for hit in hits:
            article = hit.payload.get("article_number")
            source = hit.payload.get("source", "Гражданский кодекс Российской Федерации")

            if not article:
                continue

            value = f"{source}, статья {article}"

            if value in seen:
                continue

            seen.add(value)
            sources.append(value)

        return sources