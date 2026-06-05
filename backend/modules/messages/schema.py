from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class MessageResponse(BaseModel):
    """DTO ответа с данными сообщения чата."""

    id: UUID
    chat_id: UUID
    user_id: UUID
    role: str
    content: str | None
    created_at: datetime
    chat_document_id: UUID | None = None
    analysis_result_id: UUID | None = None

    model_config = ConfigDict(
        from_attributes=True,
    )
