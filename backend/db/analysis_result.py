import uuid
from typing import Optional

from sqlalchemy.orm import Mapped, relationship

from backend.db import Message
from backend.db.base import Base, TimestampMixin


class AnalysisResult(Base, TimestampMixin):
    __tablename__ = "analysis_results"

    id: Mapped[uuid.UUID]

    summary: Mapped[str]

    risks_found: Mapped[Optional[str]]

    recommendations: Mapped[Optional[str]]

    report_file_path: Mapped[Optional[str]]

    messages: Mapped[list["Message"]] = relationship(
        back_populates="analysis_result"
    )