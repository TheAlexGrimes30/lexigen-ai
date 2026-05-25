from fastapi import APIRouter
from pydantic import Field, BaseModel

from backend.modules.rag.service import rag_app_service


class RAGQueryRequest(BaseModel):
    query: str = Field(min_length=1)


class RAGQueryResponse(BaseModel):
    answer: str


router = APIRouter(prefix="/api/rag", tags=["rag"])


@router.get("/health")
async def rag_health() -> dict:
    return rag_app_service.health()


@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(payload: RAGQueryRequest) -> RAGQueryResponse:
    answer = await rag_app_service.ask(payload.query)
    return RAGQueryResponse(answer=answer)