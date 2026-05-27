from io import BytesIO
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.modules.analytics.builders import DocxAnalysisReportBuilder
from backend.modules.analytics.interfaces import BaseAnalysisReportBuilder, BaseAnalysisResultRepository, \
    BaseAnalysisReportService
from backend.modules.analytics.repository import AnalysisResultRepository


class AnalysisReportService(BaseAnalysisReportService):
    """Сервис формирования отчётов результатов анализа."""

    def __init__(
        self,
        repository: BaseAnalysisResultRepository,
        docx_builder: BaseAnalysisReportBuilder,
    ) -> None:
        """Инициализирует сервис отчётов анализа."""
        self.repository = repository
        self.docx_builder = docx_builder

    async def build_user_docx_report(
        self,
        db: AsyncSession,
        analysis_id: UUID,
        user_id: UUID,
    ) -> BytesIO:
        """Создаёт DOCX-отчёт для результата анализа пользователя."""

        result = await self.repository.get_user_analysis_result(
            db=db,
            analysis_id=analysis_id,
            user_id=user_id,
        )

        if not result:
            raise HTTPException(
                status_code=404,
                detail="Результат анализа не найден",
            )

        return self.docx_builder.build(result.summary)

    def build_docx(
        self,
        text: str,
    ) -> BytesIO:
        """Создаёт DOCX-отчёт напрямую из текста анализа."""
        return self.docx_builder.build(text)


analysis_report_service = AnalysisReportService(
    repository=AnalysisResultRepository(),
    docx_builder=DocxAnalysisReportBuilder(),
)
