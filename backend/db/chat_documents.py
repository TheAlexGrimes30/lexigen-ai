import uuid
from typing import Optional, TYPE_CHECKING

from sqlalchemy import Text, ForeignKey, UUID, String
from sqlalchemy.orm import Mapped, relationship, mapped_column

from backend.db.base import TimestampMixin, Base

if TYPE_CHECKING:
    from backend.db.users import User
    from backend.db.messages import Message


class ChatDocument(Base, TimestampMixin):
    __tablename__ = "chat_documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    chat_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    uploaded_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    original_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    mime_type: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True
    )

    extracted_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )

    summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="chat_document",
        lazy="selectin"
    )

    uploaded_by_user: Mapped["User"] = relationship(
        lazy="selectin"
    )
