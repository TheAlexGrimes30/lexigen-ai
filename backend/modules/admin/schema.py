from pydantic import BaseModel


class AdminSubscriptionsAnalyticsResponse(BaseModel):
    total_users: int
    without_subscription: int
    by_plan: dict[str, int]