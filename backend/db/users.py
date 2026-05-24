import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import UUID, String, Enum, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db import Chat
from backend.db.base import Base, TimestampMixin


class UserRole(enum.Enum):
    user = "user"
    admin = "admin"

class UserStatus(str, enum.Enum):
    active = "active"
    banned = "banned"



class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True
    )

    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    role: Mapped[UserRole] = mapped_column(
        "primary_role",
        Enum(UserRole, name="user_role_enum", create_type=False),
        nullable=False,
        default=UserRole.user
    )

    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    chats: Mapped[list["Chat"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
