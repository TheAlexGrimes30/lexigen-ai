from datetime import datetime
from pydantic import BaseModel


class DocumentBase(BaseModel):
    title: str
    content: str


class DocumentCreate(DocumentBase):
    created_by: int | None = None


class DocumentResponse(DocumentBase):
    id: int
    created_by: int | None
    created_at: datetime

    class Config:
        from_attributes = True