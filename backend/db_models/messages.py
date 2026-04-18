import enum

from sqlalchemy import Column, Integer, Text, func, DateTime, ForeignKey, Enum as SAEnum, Index
from sqlalchemy.orm import relationship

from backend.db.base import Base

class MessageRole(enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)

    chat_id = Column(
        Integer,
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False
    )

    role = Column(
        SAEnum(MessageRole, name="message_role"),
        nullable=False
    )

    content = Column(Text, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    chat = relationship("Chat", back_populates="messages", lazy="selectin")

    __table_args__ = (
        Index("ix_messages_chat_id", "chat_id"),
    )