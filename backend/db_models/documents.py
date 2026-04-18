from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, func

from backend.db.base import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)

    title = Column(String, nullable=False)

    content = Column(Text, nullable=False)

    created_by = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )