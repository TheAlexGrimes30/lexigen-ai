from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.logger_config import get_logger
from backend.db import AnalysisResult
from backend.modules.analytics.interfaces import BaseAnalysisResultRepository

logger = get_logger(__name__)

class AnalysisResultRepository(BaseAnalysisResultRepository):
    """Репозиторий результатов анализа документов."""

    async def get_user_analysis_result(
        self,
        db: AsyncSession,
        analysis_id: UUID,
        user_id: UUID,
    ) -> AnalysisResult | None:
        """Возвращает результат анализа, принадлежащий конкретному пользователю."""

        logger.info(
            "Fetching user analysis result: analysis_id=%s, user_id=%s",
            analysis_id,
            user_id,
        )

        try:
            stmt = select(AnalysisResult).where(
                AnalysisResult.id == analysis_id,
                AnalysisResult.generated_by_user_id == user_id,
            )

            result = await db.scalar(stmt)

            logger.info(
                "User analysis result fetched: analysis_id=%s, found=%s",
                analysis_id,
                result is not None,
            )

            return result

        except Exception:
            logger.exception(
                "Failed to fetch user analysis result: analysis_id=%s, user_id=%s",
                analysis_id,
                user_id,
            )
            raise
