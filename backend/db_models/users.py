import enum
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, Enum, DateTime, func
from sqlalchemy.orm import relationship

from backend.db.base import Base


class UserRole(enum.Enum):
    USER = "user"
    ADMIN = "admin"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    role = Column(
        Enum(UserRole, name="user_role"),
        default=UserRole.USER,
        nullable=False
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    subscriptions = relationship(
        "Subscription",
        back_populates="user",
        lazy="selectin",
        cascade="all, delete-orphan"
    )

    chats = relationship(
        "Chat",
        back_populates="user",
        lazy="selectin",
        cascade="all, delete-orphan"
    )
