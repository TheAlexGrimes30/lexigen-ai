from fastapi import APIRouter

from backend.modules.rag.schemas import RAGQueryResponse, RAGQueryRequest
from backend.modules.rag.service import rag_app_service


router = APIRouter(prefix="/api/rag", tags=["rag"])


@router.get("/health")
async def rag_health() -> dict:
    return rag_app_service.health()


@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(payload: RAGQueryRequest) -> RAGQueryResponse:
    answer = await rag_app_service.ask(payload.query)
    return RAGQueryResponse(answer=answer)