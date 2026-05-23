import enum

from sqlalchemy import Integer, Column, String, DateTime, Enum, func
from sqlalchemy.orm import relationship

from backend.db.base import Base, TimestampMixin


class UserRole(enum.Enum):
    USER = "user"
    ADMIN = "admin"


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    role = Column(
        Enum(UserRole, name="user_role"),
        default=UserRole.USER,
        nullable=False
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