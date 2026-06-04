import re
from abc import ABC, abstractmethod

from llama_cpp import Llama

from backend.modules.rag.search_result_service import SearchResult


INVALID_QUERY_MESSAGE = (
    "Введите корректный юридический вопрос. "
    "Например: «Что такое акцепт оферты?», «Можно ли работать с 14 лет?», "
    "«Какие риски есть при кредите для ООО?»"
)


class BaseLLMClient(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class BasePromptBuilder(ABC):

    @abstractmethod
    def build(self, query: str, context: str) -> str:
        raise NotImplementedError


class BaseContextCleaner(ABC):

    @abstractmethod
    def clean_context(self, text: str) -> str:
        raise NotImplementedError


class BaseQueryValidator(ABC):
    """
    Interface for validating user queries before retrieval/generation.
    """

    @abstractmethod
    def validate(self, query: str) -> tuple[bool, str | None]:
        """
        Validate query and return (is_valid, error_message).
        """

        raise NotImplementedError


class BaseGenerator(ABC):

    @abstractmethod
    def generate(
            self,
            query: str,
            context: str,
            hits: list[SearchResult]
    ) -> str:
        raise NotImplementedError


class ContextCleaner(BaseContextCleaner):

    def clean_context(self, text: str) -> str:
        text = re.sub(r"#+", "", text)
        text = re.sub(r"\*+", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()


class LegalQueryValidator(BaseQueryValidator):
    """
    Lightweight guard against random/gibberish queries.

    The validator is intentionally conservative:
    - it rejects obvious garbage such as "frfref", "asdfgh", "???";
    - it allows normal Russian legal questions;
    - it allows short legal terms such as "акцепт", "договор", "статья 438";
    - it does not call external services and does not affect valid legal requests.
    """

    MIN_MEANINGFUL_CHARS = 3

    LEGAL_HINT_PATTERN = re.compile(
        r"(?iu)\b("
        r"гк|тк|коап|ук|рф|ст|статья|стать[ьяеию]+|"
        r"договор|оферта|акцепт|кредит|займ|ооо|ип|предпринимател|"
        r"должник|кредитор|обязательств|ответственност|суд|иск|"
        r"торг|аукцион|расторжен|изменен|изменён|штраф|убытк|"
        r"работ|труд|зарплат|отпуск|увольнен|несовершеннолет"
        r")\b"
    )

    ARTICLE_REFERENCE_PATTERN = re.compile(
        r"(?iu)\b(ст\.?|статья|article)\s*\d+(?:\.\d+)?\b|\b\d{2,4}(?:\.\d+)?\s*(гк|тк|коап|ук)\b"
    )

    VOWELS = set("аеёиоуыэюяaeiou")

    KEYBOARD_GIBBERISH_PATTERNS = (
        re.compile(r"(?i)^(asdf+|qwerty+|zxcv+|йцукен+|фыва+)$"),
        re.compile(r"(?i)^[a-z]{4,12}$"),
    )

    def validate(self, query: str) -> tuple[bool, str | None]:
        """
        Validate user query before LLM generation.
        """

        normalized = self._normalize(query)

        if not normalized:
            return False, INVALID_QUERY_MESSAGE

        if self._is_article_reference(normalized):
            return True, None

        if self._has_legal_hint(normalized):
            return True, None

        if self._looks_like_question(normalized) and not self._looks_like_gibberish(normalized):
            return True, None

        return False, INVALID_QUERY_MESSAGE

    @staticmethod
    def _normalize(query: str) -> str:
        """
        Normalize whitespace and strip technical noise.
        """

        query = str(query or "")
        query = re.sub(r"\s+", " ", query).strip()
        return query

    def _is_article_reference(self, query: str) -> bool:
        """
        Allow direct legal article queries such as "статья 438".
        """

        return bool(self.ARTICLE_REFERENCE_PATTERN.search(query))

    def _has_legal_hint(self, query: str) -> bool:
        """
        Detect legal-domain words that make even a short query meaningful.
        """

        return bool(self.LEGAL_HINT_PATTERN.search(query))

    def _looks_like_question(self, query: str) -> bool:
        """
        Detect a natural language question without requiring legal keywords.
        """

        letters = re.findall(r"[а-яА-ЯёЁa-zA-Z]", query)
        words = re.findall(r"[а-яА-ЯёЁa-zA-Z0-9_.-]+", query)

        if len(letters) < self.MIN_MEANINGFUL_CHARS:
            return False

        if len(words) >= 3:
            return True

        return query.endswith("?") and len(words) >= 2

    def _looks_like_gibberish(self, query: str) -> bool:
        """
        Reject obvious random input while keeping meaningful Russian words.
        """

        compact = re.sub(r"[^а-яА-ЯёЁa-zA-Z0-9]", "", query).lower()
        words = re.findall(r"[а-яА-ЯёЁa-zA-Z]+", query.lower())

        if not compact:
            return True

        if len(words) == 1:
            word = words[0]
            latin_only = bool(re.fullmatch(r"[a-z]+", word))

            if latin_only and not self._has_legal_hint(word):
                return True

            if len(word) >= 5 and sum(ch in self.VOWELS for ch in word) == 0:
                return True

        for pattern in self.KEYBOARD_GIBBERISH_PATTERNS:
            if pattern.fullmatch(compact):
                return True

        repeated_ratio = self._max_repeated_char_ratio(compact)
        if len(compact) >= 5 and repeated_ratio >= 0.75:
            return True

        return False

    @staticmethod
    def _max_repeated_char_ratio(text: str) -> float:
        """
        Return max frequency ratio for one character.
        """

        if not text:
            return 1.0

        return max(text.count(char) for char in set(text)) / len(text)


class QwenClient(BaseLLMClient):

    def __init__(self, model_path: str, n_ctx: int = 2048, max_tokens: int = 400):
        self.max_tokens = max_tokens
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=8,
            verbose=False
        )

    def generate(self, prompt: str) -> str:
        output = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты модуль извлечения юридических норм.\n"
                        "НЕ веди диалог.\n"
                        "НЕ объясняй процесс.\n"
                        "НЕ используй слова типа: 'сначала', 'проверяю', 'анализирую'.\n"
                        "Выводи только готовый юридический ответ.\n"
                        "Никаких рассуждений и пояснений."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.15,
            top_p=0.85,
            repeat_penalty=1.1,
            max_tokens=self.max_tokens
        )

        return output["choices"][0]["message"]["content"].strip()


class CreditPromptBuilder(BasePromptBuilder):

    def build(self, query: str, context: str) -> str:
        return f"""
        Ты извлекаешь юридический ответ ТОЛЬКО из контекста.

        ПРАВИЛА:
        - НЕ объясняй ход мыслей
        - НЕ используй слова: "сначала", "проверяю", "анализ"
        - НЕ добавляй внешние знания
        - НЕ рассуждай

        ФОРМАТ:
        - 3–6 предложений
        - юридически точный текст
        - без списков
        - без вступлений

        ЕСЛИ НЕТ ДАННЫХ:
        Ответ: "Нет данных в предоставленных источниках"

        =====================
        КОНТЕКСТ
        =====================
        {context}

        =====================
        ВОПРОС
        =====================
        {query}

        =====================
        ОТВЕТ:
        =====================
        """.strip()


class ContractRiskAnalysisPromptBuilder(BasePromptBuilder):

    def build(self, query: str, context: str) -> str:
        return f"""
        Проанализируй договор по нормам ГК РФ из контекста.

        Пиши кратко. Ответ строго в 4 строки. Без длинных объяснений.

        Формат:
        1. Вывод: ...
        2. Риски: ...
        3. Слабые условия: ...
        4. Рекомендации: ...

        КОНТЕКСТ:
        {context}

        ДОГОВОР:
        {query}

        ОТВЕТ:
        """.strip()


class Generator(BaseGenerator):

    def __init__(self, llm, prompt_builder, cleaner, query_validator: BaseQueryValidator | None = None):
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.cleaner = cleaner
        self.query_validator = query_validator or LegalQueryValidator()

    def generate(
            self,
            query: str,
            context: str,
            hits: list[SearchResult]
    ) -> str:

        is_valid, error_message = self.query_validator.validate(query)
        if not is_valid:
            return error_message or INVALID_QUERY_MESSAGE

        context = self.cleaner.clean_context(context or "")

        if len(context) < 80:
            context = self._build_fallback_context(hits)

        if len(context) < 30:
            return "Недостаточно данных."

        prompt = self.prompt_builder.build(query, context)

        try:
            raw = self.llm.generate(prompt)
        except Exception as e:
            print(f"[GENERATION ERROR] {e}")
            return "Ошибка генерации ответа."

        return self._postprocess(raw)

    def _build_fallback_context(self, hits: list[SearchResult]) -> str:
        parts = []

        for h in hits[:5]:
            text = (h.text or "").strip()
            if len(text) < 20:
                continue

            article = h.payload.get("article_number", "?")
            header = h.payload.get("header", "")

            parts.append(f"Статья {article} — {header}\n{text[:700]}")

        return "\n\n".join(parts)

    def _postprocess(self, text: str) -> str:

        if not text:
            return "Недостаточно данных."

        text = re.sub(r"<.*?>", "", text).strip()

        text = re.sub(r"(?i)^(a|answer|ответ):\s*", "", text)

        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"[ \t]+", " ", text).strip()

        if len(text.split()) < 6:
            return "Недостаточно данных."

        if re.search(r"(?i)\b(сначала|проверяю|анализирую|рассмотрю)\b", text):
            text = re.sub(r"(?i)\b(сначала|проверяю|анализирую|рассмотрю).*", "", text).strip()

        return text

    def close(self):
        try:
            self.llm.close()
        except Exception:
            pass
