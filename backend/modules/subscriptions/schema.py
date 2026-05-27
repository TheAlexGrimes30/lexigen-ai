from pydantic import BaseModel, Field

from backend.db import SubscriptionPlan


class SubscriptionPlanResponse(BaseModel):
    """DTO ответа с данными тарифного плана."""

    plan: SubscriptionPlan
    title: str
    price_rub: int
    description: str


class SubscriptionUpdateRequest(BaseModel):
    """DTO запроса изменения подписки пользователя."""

    plan: SubscriptionPlan = Field(description="Выбранный тариф")


class SubscriptionResponse(BaseModel):
    """DTO ответа с текущей подпиской пользователя."""

    plan: SubscriptionPlan | None
    title: str
    price_rub: int
    is_active: bool
    can_analyze_unlimited: bool
