from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import get_db
from backend.db.users import User
from backend.modules.analytics.service import analysis_report_service
from backend.modules.auth.dependencies import get_current_user

router = APIRouter(
    prefix="/api/analysis-results",
    tags=["analysis-results"],
)


@router.get("/{analysis_id}/download")
async def download_analysis_result(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Скачивает DOCX-отчёт результата анализа текущего пользователя."""

    buffer = await analysis_report_service.build_user_docx_report(
        db=db,
        analysis_id=analysis_id,
        user_id=current_user.id,
    )

    return StreamingResponse(
        buffer,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
        headers={
            "Content-Disposition": (
                "attachment; filename=analysis_result.docx"
            ),
        },
    )
