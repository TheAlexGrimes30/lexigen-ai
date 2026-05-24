from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import get_db
from backend.db.users import User
from backend.modules.auth.dependencies import get_current_user
from backend.modules.chats.schema import ChatCreateRequest, ChatResponse
from backend.modules.chats.service import chats_service

router = APIRouter(prefix="/api/chats", tags=["chats"])


@router.get("", response_model=list[ChatResponse])
async def get_chats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await chats_service.list_chats_response(db, current_user)


@router.post("", response_model=ChatResponse)
async def post_chat(
    payload: ChatCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await chats_service.create_chat_response(db, payload, current_user)


@router.delete("/{chat_id}")
async def delete_chat(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await chats_service.delete_chat_response(db, chat_id, current_user)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Чат не найден")
    return result
