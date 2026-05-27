from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import User
from backend.db.database import get_db
from backend.modules.auth.dependencies import get_current_user
from backend.modules.subscriptions.schema import (
    SubscriptionPlanResponse,
    SubscriptionResponse,
    SubscriptionUpdateRequest,
)
from backend.modules.subscriptions.service import subscriptions_service

router = APIRouter(
    prefix="/api/subscriptions",
    tags=["subscriptions"],
)


@router.get(
    "/plans",
    response_model=list[SubscriptionPlanResponse],
)
async def get_subscription_plans():
    """Возвращает список доступных тарифных планов."""
    return subscriptions_service.list_plans()


@router.get(
    "/me",
    response_model=SubscriptionResponse,
)
async def get_my_subscription(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Возвращает текущую подписку пользователя."""
    return await subscriptions_service.get_user_subscription_response(
        db=db,
        user=current_user,
    )


@router.put(
    "/me",
    response_model=SubscriptionResponse,
)
async def put_my_subscription(
    payload: SubscriptionUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Изменяет текущую подписку пользователя."""
    return await subscriptions_service.set_user_subscription(
        db=db,
        user=current_user,
        plan=payload.plan,
    )
