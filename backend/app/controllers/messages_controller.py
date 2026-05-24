from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.schemas.message import MessageCreateRequest, MessageResponse
from backend.modules.chats.service import chats_service
from backend.modules.messages.service import messages_service


class MessagesController:
    async def list_messages(self, db: AsyncSession, chat_id: UUID) -> list[MessageResponse]:
        messages = await messages_service.list_messages(db, chat_id)
        return [MessageResponse.model_validate(message) for message in messages]

    async def create_message_turn(self, db: AsyncSession, chat_id: UUID, payload: MessageCreateRequest) -> list[MessageResponse]:
        chat = await chats_service.get_chat(db, chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Чат не найден")

        user_message, assistant_message = await messages_service.create_dialog_turn(
            db=db,
            chat_id=chat_id,
            user_id=chat.user_id,
            user_text=payload.content,
        )
        return [
            MessageResponse.model_validate(user_message),
            MessageResponse.model_validate(assistant_message),
        ]

    async def create_system_error(self, db: AsyncSession, chat_id: UUID, error_text: str) -> MessageResponse:
        chat = await chats_service.get_chat(db, chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Чат не найден")

        system_message = await messages_service.create_system_error(
            db=db,
            chat_id=chat_id,
            user_id=chat.user_id,
            error_text=error_text,
        )
        return MessageResponse.model_validate(system_message)


messages_controller = MessagesController()
