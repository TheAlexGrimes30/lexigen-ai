from datetime import datetime
from enum import Enum
from pydantic import BaseModel


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class MessageBase(BaseModel):
    chat_id: int
    role: MessageRole
    content: str


class MessageCreate(MessageBase):
    pass


class MessageResponse(MessageBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True