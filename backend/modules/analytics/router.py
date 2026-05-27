from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.analysis_result import AnalysisResult
from backend.db.database import get_db
from backend.db.users import User
from backend.modules.analytics.service import analysis_report_service
from backend.modules.auth.dependencies import get_current_user

router = APIRouter(prefix="/api/analysis-results", tags=["analysis-results"])


@router.get("/{analysis_id}/download")
async def download_analysis_result(
        analysis_id: UUID,
        format: str = Query(default="docx", pattern="^(docx|pdf)$"),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ) -> StreamingResponse:

    stmt = select(AnalysisResult).where(AnalysisResult.id == analysis_id, AnalysisResult.generated_by_user_id == current_user.id)
    result = await db.scalar(stmt)
    if not result:
        raise HTTPException(status_code=404, detail="Результат анализа не найден")
    if format == "pdf":
        buffer = analysis_report_service.build_pdf(result.summary)
        return StreamingResponse(buffer, media_type="application/pdf",
                                 headers={"Content-Disposition": "attachment; filename=analysis_result.pdf"})

    buffer = analysis_report_service.build_docx(result.summary)
    return StreamingResponse(buffer,
                             media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                             headers={"Content-Disposition": "attachment; filename=analysis_result.docx"})
