import asyncio
from enum import Enum
from typing import Optional

from backend.modules.rag.rag_engine import RAG


class RAGStatus(str, Enum):
    NOT_STARTED = "not_started"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class RAGApplicationService:

    def __init__(self):
        self._rag: Optional[RAG] = None
        self._status: RAGStatus = RAGStatus.NOT_STARTED
        self._error: str | None = None
        self._lock = asyncio.Lock()

    @property
    def status(self) -> RAGStatus:
        return self._status

    @property
    def error(self) -> str | None:
        return self._error

    async def startup(self) -> None:
        if self._status in {RAGStatus.LOADING, RAGStatus.READY}:
            return

        self._status = RAGStatus.LOADING
        self._error = None

        asyncio.create_task(self._load())

    async def _load(self) -> None:
        async with self._lock:
            try:
                rag = await asyncio.to_thread(RAG)
                await asyncio.to_thread(rag.build_and_index)

                self._rag = rag
                self._status = RAGStatus.READY
                self._error = None

            except Exception as exc:
                self._rag = None
                self._status = RAGStatus.ERROR
                self._error = str(exc)

                print(f"[RAG STARTUP ERROR] {exc}")

    async def ask(self, query: str) -> str:
        query = (query or "").strip()

        if not query:
            return "Пустой запрос. Напишите вопрос, чтобы я смог найти ответ."

        self._ensure_ready()

        try:
            return await asyncio.to_thread(
                self._rag.ask,
                query,
            )
        except Exception as exc:
            print(f"[RAG ASK ERROR] {exc}")
            raise RuntimeError(f"Ошибка при генерации RAG-ответа: {exc}") from exc

    async def analyze_contract(self, contract_text: str) -> str:
        contract_text = (contract_text or "").strip()

        if not contract_text:
            return "Не удалось извлечь текст из документа."

        self._ensure_ready()

        try:
            return await asyncio.to_thread(
                self._rag.analyze_contract,
                contract_text,
            )
        except Exception as exc:
            print(f"[RAG DOCUMENT ANALYSIS ERROR] {exc}")
            raise RuntimeError(f"Ошибка при анализе документа: {exc}") from exc

    def _ensure_ready(self) -> None:
        if self._status == RAGStatus.LOADING:
            raise RuntimeError("RAG ещё загружается. Попробуйте отправить вопрос чуть позже.")

        if self._status == RAGStatus.ERROR:
            raise RuntimeError(f"RAG недоступен: {self._error}")

        if self._rag is None:
            raise RuntimeError("RAG ещё не инициализирован.")

    def health(self) -> dict:
        return {
            "status": self._status.value,
            "ready": self._status == RAGStatus.READY,
            "error": self._error,
        }


rag_app_service = RAGApplicationService()