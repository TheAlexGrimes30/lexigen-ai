import asyncio
from enum import Enum
from typing import Optional

from backend.app.logger_config import get_logger
from backend.modules.rag.rag_engine import RAG


logger = get_logger(__name__)


class RAGStatus(str, Enum):
    NOT_STARTED = "not_started"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class RAGApplicationService:
    """Сервис жизненного цикла RAG-движка."""

    def __init__(self) -> None:
        """Инициализирует состояние сервиса RAG."""

        self._rag: Optional[RAG] = None
        self._status: RAGStatus = RAGStatus.NOT_STARTED
        self._error: str | None = None
        self._lock = asyncio.Lock()

    @property
    def status(self) -> RAGStatus:
        """Возвращает текущий статус RAG."""

        return self._status

    @property
    def error(self) -> str | None:
        """Возвращает текст последней ошибки."""

        return self._error

    async def startup(self) -> None:
        """Запускает фоновую инициализацию RAG."""

        if self._status in {
            RAGStatus.LOADING,
            RAGStatus.READY,
        }:
            logger.info(
                "RAG startup skipped: current_status=%s",
                self._status.value,
            )
            return

        logger.info("RAG startup initiated")

        self._status = RAGStatus.LOADING
        self._error = None

        asyncio.create_task(self._load())

    async def _load(self) -> None:
        """Загружает и индексирует RAG-движок."""

        async with self._lock:

            logger.info("RAG loading started")

            try:
                logger.info("Creating RAG instance")

                rag = await asyncio.to_thread(RAG)

                logger.info("Building RAG index")

                await asyncio.to_thread(
                    rag.build_and_index,
                )

                self._rag = rag
                self._status = RAGStatus.READY
                self._error = None

                logger.info(
                    "RAG loaded successfully and is ready"
                )

            except Exception as exc:
                self._rag = None
                self._status = RAGStatus.ERROR
                self._error = str(exc)

                logger.exception(
                    "RAG startup failed"
                )

    async def ask(
        self,
        query: str,
    ) -> str:
        """Возвращает ответ RAG на пользовательский запрос."""

        query = (query or "").strip()

        if not query:
            logger.warning(
                "Empty RAG query received"
            )

            return (
                "Пустой запрос. "
                "Напишите вопрос, чтобы я смог найти ответ."
            )

        logger.info(
            "RAG query received: length=%s",
            len(query),
        )

        self._ensure_ready()

        try:
            response = await asyncio.to_thread(
                self._rag.ask,
                query,
            )

            logger.info(
                "RAG answer generated successfully"
            )

            return response

        except Exception as exc:
            logger.exception(
                "RAG answer generation failed"
            )

            raise RuntimeError(
                f"Ошибка при генерации RAG-ответа: {exc}"
            ) from exc

    async def analyze_contract(
        self,
        contract_text: str,
    ) -> str:
        """Выполняет анализ договора через RAG."""

        contract_text = (
            contract_text or ""
        ).strip()

        if not contract_text:
            logger.warning(
                "Empty contract text received"
            )

            return (
                "Не удалось извлечь текст из документа."
            )

        logger.info(
            "Contract analysis requested: length=%s",
            len(contract_text),
        )

        self._ensure_ready()

        try:
            result = await asyncio.to_thread(
                self._rag.analyze_contract,
                contract_text,
            )

            logger.info(
                "Contract analysis completed successfully"
            )

            return result

        except Exception as exc:
            logger.exception(
                "Contract analysis failed"
            )

            raise RuntimeError(
                f"Ошибка при анализе документа: {exc}"
            ) from exc

    def _ensure_ready(self) -> None:
        """Проверяет готовность RAG к работе."""

        if self._status == RAGStatus.LOADING:

            logger.warning(
                "RAG requested while loading"
            )

            raise RuntimeError(
                "RAG ещё загружается. "
                "Попробуйте отправить вопрос чуть позже."
            )

        if self._status == RAGStatus.ERROR:

            logger.warning(
                "RAG requested while unavailable: error=%s",
                self._error,
            )

            raise RuntimeError(
                f"RAG недоступен: {self._error}"
            )

        if self._rag is None:

            logger.warning(
                "RAG requested before initialization"
            )

            raise RuntimeError(
                "RAG ещё не инициализирован."
            )

    def health(self) -> dict:
        """Возвращает информацию о состоянии RAG."""

        logger.debug(
            "RAG health requested: status=%s",
            self._status.value,
        )

        return {
            "status": self._status.value,
            "ready": self._status == RAGStatus.READY,
            "error": self._error,
        }


rag_app_service = RAGApplicationService()