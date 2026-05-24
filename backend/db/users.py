import uuid
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from sqlalchemy import UUID, String, Enum, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship


from backend.db.base import Base, TimestampMixin
from backend.db.enums import UserRole

if TYPE_CHECKING:
    from backend.db.chats import Chat


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

    subscriptions = relationship(
        "Subscription",
        back_populates="user",
        cascade="all, delete-orphan"
    )
