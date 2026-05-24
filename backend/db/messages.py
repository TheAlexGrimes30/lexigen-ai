import enum
import uuid
from typing import Optional

from sqlalchemy import UUID, ForeignKey, Enum, Text, Index
from sqlalchemy.orm import mapped_column, Mapped, relationship

from backend.db import Chat
from backend.db.base import Base


class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"
    system = "system"

class MessageType(str, enum.Enum):
    text = "text"
    document = "document"
    analysis = "analysis"


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    chat_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    role: Mapped[MessageRole] = mapped_column(
        Enum(MessageRole, name="message_role_enum"),
        nullable=False
    )

    type: Mapped[MessageType] = mapped_column(
        Enum(MessageType, name="message_type_enum"),
        nullable=False,
        default=MessageType.text
    )

    content: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )

    chat: Mapped["Chat"] = relationship(
        back_populates="messages"
    )

    __table_args__ = (
        Index("ix_messages_chat_id", "chat_id"),
    )
