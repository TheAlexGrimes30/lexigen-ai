from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Form, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import get_db
from backend.db.users import User
from backend.modules.auth.dependencies import get_current_user
from backend.modules.messages.schema import MessageResponse
from backend.modules.messages.service import messages_service

router = APIRouter(prefix="/api/chats/{chat_id}/messages", tags=["messages"])


@router.get("", response_model=list[MessageResponse])
async def get_messages(chat_id: UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    return await messages_service.list_messages_response(db, chat_id, current_user)


@router.post("", response_model=list[MessageResponse])
async def post_message(
    chat_id: UUID,
    content: str = Form(""),
    file: UploadFile | None = File(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return await messages_service.create_message_turn_response(db=db, chat_id=chat_id, content=content, file=file, current_user=current_user)
    except Exception as exc:
        try:
            system_message = await messages_service.create_system_error_response(db, chat_id, f"Системная ошибка при обработке сообщения: {exc}", current_user)
            return [system_message]
        except Exception:
            raise HTTPException(status_code=500, detail="Не удалось обработать сообщение") from exc