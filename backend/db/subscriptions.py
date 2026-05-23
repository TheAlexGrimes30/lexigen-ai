from sqlalchemy import Integer, Column, ForeignKey, String, Boolean, DateTime
from sqlalchemy.orm import relationship

from backend.db.base import Base, TimestampMixin


class Subscription(Base, TimestampMixin):
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True)

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )

    plan_name = Column(String, nullable=False)

    is_active = Column(Boolean, default=True)

    expires_at = Column(DateTime(timezone=True))

    user = relationship("User", back_populates="subscriptions")