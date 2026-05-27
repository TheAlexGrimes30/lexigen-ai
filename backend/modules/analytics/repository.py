from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import AnalysisResult
from backend.modules.analytics.interfaces import BaseAnalysisResultRepository


class AnalysisResultRepository(BaseAnalysisResultRepository):
    """Репозиторий результатов анализа документов."""

    async def get_user_analysis_result(
        self,
        db: AsyncSession,
        analysis_id: UUID,
        user_id: UUID,
    ) -> AnalysisResult | None:
        """Возвращает результат анализа, принадлежащий конкретному пользователю."""

        stmt = select(AnalysisResult).where(
            AnalysisResult.id == analysis_id,
            AnalysisResult.generated_by_user_id == user_id,
        )

        return await db.scalar(stmt)
