from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.modules.messages.schema import MessageCreateRequest, MessageResponse
from backend.modules.messages.service import messages_service
from backend.db.database import get_db

router = APIRouter(prefix="/api/chats/{chat_id}/messages", tags=["messages"])


@router.get("", response_model=list[MessageResponse])
async def get_messages(chat_id: UUID, db: AsyncSession = Depends(get_db)):
    return await messages_service.list_messages_response(db, chat_id)


@router.post("", response_model=list[MessageResponse])
async def post_message(chat_id: UUID, payload: MessageCreateRequest, db: AsyncSession = Depends(get_db)):
    try:
        return await messages_service.create_message_turn_response(db, chat_id, payload)
    except Exception as exc:
        try:
            system_message = await messages_service.create_system_error_response(
                db,
                chat_id,
                f"Системная ошибка при обработке сообщения: {exc}",
            )
            return [system_message]
        except Exception:
            raise HTTPException(status_code=500, detail="Не удалось обработать сообщение") from exc
