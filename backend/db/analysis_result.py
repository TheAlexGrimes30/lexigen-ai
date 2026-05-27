import uuid
from typing import Optional, TYPE_CHECKING

from sqlalchemy import UUID, ForeignKey, Text, String
from sqlalchemy.orm import Mapped, relationship, mapped_column

from backend.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.db.users import User
    from backend.db.messages import Message


class AnalysisResult(Base, TimestampMixin):
    __tablename__ = "analysis_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    generated_by_user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    summary: Mapped[str] = mapped_column(Text, nullable=False)
    risks_found: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    recommendations: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    messages: Mapped[list["Message"]] = relationship(back_populates="analysis_result", lazy="selectin")
    generated_by_user: Mapped["User"] = relationship(lazy="selectin")