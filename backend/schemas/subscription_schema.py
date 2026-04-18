from datetime import datetime
from pydantic import BaseModel


class SubscriptionBase(BaseModel):
    user_id: int
    plan_name: str
    is_active: bool = True


class SubscriptionCreate(SubscriptionBase):
    expires_at: datetime | None = None


class SubscriptionResponse(SubscriptionBase):
    id: int
    expires_at: datetime | None

    class Config:
        from_attributes = True