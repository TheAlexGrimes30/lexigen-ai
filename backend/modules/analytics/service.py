from io import BytesIO
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.logger_config import get_logger
from backend.modules.analytics.builders import DocxAnalysisReportBuilder
from backend.modules.analytics.interfaces import BaseAnalysisReportBuilder, BaseAnalysisResultRepository, \
    BaseAnalysisReportService
from backend.modules.analytics.repository import AnalysisResultRepository

logger = get_logger(__name__)

class AnalysisReportService(BaseAnalysisReportService):
    """Сервис формирования отчётов результатов анализа."""

    def __init__(
        self,
        repository: BaseAnalysisResultRepository,
        docx_builder: BaseAnalysisReportBuilder,
    ):
        self.repository = repository
        self.docx_builder = docx_builder

    async def build_user_docx_report(
        self,
        db: AsyncSession,
        analysis_id: UUID,
        user_id: UUID,
    ) -> BytesIO:
        """Создаёт DOCX-отчёт для результата анализа пользователя."""

        logger.info(
            "User DOCX analysis report generation started: analysis_id=%s, user_id=%s",
            analysis_id,
            user_id,
        )

        try:
            result = await self.repository.get_user_analysis_result(
                db=db,
                analysis_id=analysis_id,
                user_id=user_id,
            )

            if not result:
                logger.warning(
                    "Analysis result not found for DOCX report: analysis_id=%s, user_id=%s",
                    analysis_id,
                    user_id,
                )

                raise HTTPException(
                    status_code=404,
                    detail="Результат анализа не найден",
                )

            report = self.docx_builder.build(result.summary)

            logger.info(
                "User DOCX analysis report generated successfully: analysis_id=%s, user_id=%s",
                analysis_id,
                user_id,
            )

            return report

        except HTTPException:
            raise

        except Exception:
            logger.exception(
                "Failed to generate user DOCX analysis report: analysis_id=%s, user_id=%s",
                analysis_id,
                user_id,
            )
            raise

    def build_docx(
        self,
        text: str,
    ) -> BytesIO:
        """Создаёт DOCX-отчёт напрямую из текста анализа."""

        logger.info("Direct DOCX analysis report generation started")

        try:
            report = self.docx_builder.build(text)

            logger.info("Direct DOCX analysis report generated successfully")

            return report

        except Exception:
            logger.exception("Failed to generate direct DOCX analysis report")
            raise


analysis_report_service = AnalysisReportService(
    repository=AnalysisResultRepository(),
    docx_builder=DocxAnalysisReportBuilder(),
)
