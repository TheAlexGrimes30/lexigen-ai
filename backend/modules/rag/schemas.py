from pydantic import BaseModel, Field


class RAGQueryRequest(BaseModel):
    query: str = Field(min_length=1)


class RAGQueryResponse(BaseModel):
    answer: str