from fastapi import Depends, HTTPException, APIRouter
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import User, UserRole
from backend.db.database import get_db
from backend.modules.auth.dependencies import get_current_user
from backend.modules.subscriptions.schema import SubscriptionResponse, SubscriptionUpdateRequest, \
    SubscriptionAnalyticsResponse, SubscriptionPlanResponse
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
    return subscriptions_service.list_plans()


@router.get(
    "/me",
    response_model=SubscriptionResponse,
)
async def get_my_subscription(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await subscriptions_service.get_user_subscription_response(
        db,
        current_user,
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
    return await subscriptions_service.set_user_subscription(
        db,
        current_user,
        payload.plan,
    )


@router.get(
    "/admin/analytics",
    response_model=SubscriptionAnalyticsResponse,
)
async def get_subscription_admin_analytics(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != UserRole.admin:
        raise HTTPException(
            status_code=403,
            detail="Доступ разрешён только администратору",
        )

    return await subscriptions_service.get_admin_analytics(db)
