from sqlalchemy import ForeignKey, Integer, Column, DateTime, func
from sqlalchemy.orm import relationship

from backend.db.base import Base

class Chat(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True)

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    user = relationship("User", back_populates="chats")

    messages = relationship(
        "Message",
        back_populates="chat",
        lazy="selectin",
        cascade="all, delete-orphan"
    )
