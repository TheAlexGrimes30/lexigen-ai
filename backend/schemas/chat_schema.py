from datetime import datetime
from pydantic import BaseModel


class ChatBase(BaseModel):
    user_id: int


class ChatCreate(ChatBase):
    pass


class ChatResponse(ChatBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True