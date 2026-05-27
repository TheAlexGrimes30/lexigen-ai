from abc import abstractmethod, ABC
from io import BytesIO
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import AnalysisResult


class BaseAnalysisResultRepository(ABC):
    """Абстрактный репозиторий результатов анализа."""

    @abstractmethod
    async def get_user_analysis_result(
        self,
        db: AsyncSession,
        analysis_id: UUID,
        user_id: UUID,
    ) -> AnalysisResult | None:
        """Возвращает результат анализа пользователя по идентификатору."""

        raise NotImplementedError


class BaseAnalysisReportBuilder(ABC):
    """Абстрактный генератор файлов отчёта анализа."""

    @abstractmethod
    def build(
        self,
        text: str,
    ) -> BytesIO:
        """Создаёт файл отчёта из текста анализа."""
        raise NotImplementedError


class BaseAnalysisReportService(ABC):
    """Абстрактный сервис формирования отчётов анализа."""

    @abstractmethod
    async def build_user_docx_report(
        self,
        db: AsyncSession,
        analysis_id: UUID,
        user_id: UUID,
    ) -> BytesIO:
        """Создаёт DOCX-отчёт результата анализа пользователя."""

        raise NotImplementedError
