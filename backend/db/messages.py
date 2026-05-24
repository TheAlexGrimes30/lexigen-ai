import enum
import uuid
from typing import Optional

from sqlalchemy import UUID, ForeignKey, Enum, Text, Index
from sqlalchemy.orm import mapped_column, Mapped, relationship

from backend.db import Chat
from backend.db.base import Base, TimestampMixin
from backend.db.chat_documents import ChatDocument


class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class Message(Base, TimestampMixin):
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

    content: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )

    chat_document_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_documents.id", ondelete="SET NULL"),
        nullable=True
    )

    analysis_result_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_results.id", ondelete="SET NULL"),
        nullable=True
    )

    chat: Mapped["Chat"] = relationship(
        back_populates="messages"
    )

    chat_document: Mapped[Optional["ChatDocument"]] = relationship(
        back_populates="messages",
        lazy="selectin"
    )

    analysis_result: Mapped[Optional["AnalysisResult"]] = relationship(
        back_populates="messages",
        lazy="selectin"
    )

    __table_args__ = (
        Index("ix_messages_chat_id", "chat_id"),
    )