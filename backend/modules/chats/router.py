from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.modules.chats.schema import ChatCreateRequest, ChatResponse
from backend.modules.chats.service import chats_service
from backend.db.database import get_db

router = APIRouter(prefix="/api/chats", tags=["chats"])


@router.get("", response_model=list[ChatResponse])
async def get_chats(db: AsyncSession = Depends(get_db)):
    return await chats_service.list_chats_response(db)


@router.post("", response_model=ChatResponse)
async def post_chat(payload: ChatCreateRequest, db: AsyncSession = Depends(get_db)):
    return await chats_service.create_chat_response(db, payload)


@router.delete("/{chat_id}")
async def delete_chat(chat_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await chats_service.delete_chat_response(db, chat_id)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Чат не найден")
    return result
