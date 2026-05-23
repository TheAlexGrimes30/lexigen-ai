from sqlalchemy import Integer, Column, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship

from backend.db.base import Base, TimestampMixin


class Chat(Base, TimestampMixin):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True)

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )

    user = relationship("User", back_populates="chats")

    messages = relationship(
        "Message",
        back_populates="chat",
        lazy="selectin",
        cascade="all, delete-orphan"
    )