from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class ChatCreateRequest(BaseModel):
    """DTO запроса создания чата."""

    title: str = Field(min_length=1, max_length=255)


class ChatUpdateRequest(BaseModel):
    """DTO запроса изменения названия чата."""

    title: str = Field(min_length=1, max_length=255)


class ChatResponse(BaseModel):
    """DTO ответа с данными чата."""

    model_config = ConfigDict(
        from_attributes=True,
    )

    id: UUID
    user_id: UUID
    title: str
    created_at: datetime

