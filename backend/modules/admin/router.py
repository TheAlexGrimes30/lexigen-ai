from fastapi import APIRouter
from fastapi.params import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import User
from backend.db.database import get_db
from backend.modules.admin.schema import AdminSubscriptionsAnalyticsResponse
from backend.modules.admin.service import admin_service
from backend.modules.auth.dependencies import get_admin_user

router = APIRouter(
    prefix="/api/admin",
    tags=["admin"],
)


@router.get(
    "/analytics",
    response_model=AdminSubscriptionsAnalyticsResponse,
)
async def get_admin_analytics(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_admin_user),
):
    """Возвращает административную аналитику по пользователям и подпискам."""
    return await admin_service.get_users_by_subscription_analytics(db)
