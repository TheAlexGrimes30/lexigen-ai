from pydantic import BaseModel, Field

from backend.db import SubscriptionPlan


class SubscriptionPlanResponse(BaseModel):
    plan: SubscriptionPlan
    title: str
    price_rub: int
    description: str


class SubscriptionUpdateRequest(BaseModel):
    plan: SubscriptionPlan = Field(description="Выбранный тариф")


class SubscriptionResponse(BaseModel):
    plan: SubscriptionPlan | None
    title: str
    price_rub: int
    is_active: bool
    can_analyze_unlimited: bool
