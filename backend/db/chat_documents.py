import uuid
from typing import Optional

from sqlalchemy.orm import Mapped, relationship

from backend.db import Message
from backend.db.base import TimestampMixin, Base


class ChatDocument(Base, TimestampMixin):
    __tablename__ = "chat_documents"

    id: Mapped[uuid.UUID]

    filename: Mapped[str]

    file_path: Mapped[str]

    extracted_text: Mapped[Optional[str]]

    messages: Mapped[list["Message"]] = relationship(
        back_populates="chat_document"
    )