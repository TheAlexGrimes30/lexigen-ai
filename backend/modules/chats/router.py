from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.controllers.chats_controller import chats_controller
from backend.app.schemas.chat import ChatCreateRequest, ChatResponse
from backend.db.database import get_db

router = APIRouter(prefix="/api/chats", tags=["chats"])


@router.get("", response_model=list[ChatResponse])
async def get_chats(db: AsyncSession = Depends(get_db)):
    return await chats_controller.list_chats(db)


@router.post("", response_model=ChatResponse)
async def post_chat(payload: ChatCreateRequest, db: AsyncSession = Depends(get_db)):
    return await chats_controller.create_chat(db, payload)
