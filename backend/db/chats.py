import uuid

from sqlalchemy import UUID, String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db import User, Message
from backend.db.analysis_result import AnalysisResult
from backend.db.base import Base, TimestampMixin
from backend.db.chat_documents import ChatDocument


class Chat(Base, TimestampMixin):
    __tablename__ = "chats"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    user: Mapped["User"] = relationship(
        back_populates="chats"
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="chat",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="Message.created_at"
    )

    documents: Mapped[list["ChatDocument"]] = relationship(
        cascade="all, delete-orphan",
        lazy="selectin"
    )

    analysis_results: Mapped[list["AnalysisResult"]] = relationship(
        cascade="all, delete-orphan",
        lazy="selectin"
    )

