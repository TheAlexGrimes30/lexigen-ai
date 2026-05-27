from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ChatCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)

class ChatUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)

class ChatResponse(BaseModel):
    id: UUID
    user_id: UUID
    title: str
    created_at: datetime

    class Config:
        from_attributes = True
