from sqlalchemy import Column, Integer, ForeignKey, String, Boolean, DateTime
from sqlalchemy.orm import relationship

from backend.db.base import Base

class Subscription(Base):
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