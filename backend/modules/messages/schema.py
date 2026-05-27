from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class MessageCreateRequest(BaseModel):
    content: str = Field(min_length=1)


class MessageResponse(BaseModel):
    id: UUID
    chat_id: UUID
    user_id: UUID
    role: str
    content: str | None
    created_at: datetime
    chat_document_id: UUID | None = None
    analysis_result_id: UUID | None = None

    class Config:
        from_attributes = True
