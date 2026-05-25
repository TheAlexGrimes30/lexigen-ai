from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.enums import MessageRole
from backend.db.messages import Message
from backend.db.users import User
from backend.modules.chats.service import chats_service
from backend.modules.messages.schema import MessageCreateRequest, MessageResponse
from backend.modules.rag.service import rag_app_service


class MessagesService:
    async def list_messages(self, db: AsyncSession, chat_id: UUID) -> list[Message]:
        stmt = select(Message).where(Message.chat_id == chat_id).order_by(Message.created_at.asc())
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def create_dialog_turn(self, db: AsyncSession, chat_id: UUID, user_id: UUID, user_text: str) -> tuple[Message, Message]:
        user_message = Message(
            chat_id=chat_id,
            user_id=user_id,
            role=MessageRole.user,
            content=user_text,
        )
        db.add(user_message)
        await db.flush()

        try:
            assistant_text = await rag_app_service.ask(user_text)
            assistant_role = MessageRole.assistant
        except Exception as exc:
            assistant_text = f"Система не смогла получить RAG-ответ: {exc}"
            assistant_role = MessageRole.system

        assistant_message = Message(
            chat_id=chat_id,
            user_id=user_id,
            role=assistant_role,
            content=assistant_text,
        )
        db.add(assistant_message)

        await db.commit()
        await db.refresh(user_message)
        await db.refresh(assistant_message)
        return user_message, assistant_message

    async def create_system_error(self, db: AsyncSession, chat_id: UUID, user_id: UUID, error_text: str) -> Message:
        system_message = Message(
            chat_id=chat_id,
            user_id=user_id,
            role=MessageRole.system,
            content=error_text,
        )
        db.add(system_message)
        await db.commit()
        await db.refresh(system_message)
        return system_message

    async def list_messages_response(self, db: AsyncSession, chat_id: UUID, current_user: User) -> list[MessageResponse]:
        chat = await chats_service.get_chat(db, chat_id, current_user)
        if not chat:
            raise HTTPException(status_code=404, detail="Чат не найден")

        messages = await self.list_messages(db, chat_id)
        return [MessageResponse.model_validate(message) for message in messages]

    async def create_message_turn_response(
        self,
        db: AsyncSession,
        chat_id: UUID,
        payload: MessageCreateRequest,
        current_user: User,
    ) -> list[MessageResponse]:
        chat = await chats_service.get_chat(db, chat_id, current_user)
        if not chat:
            raise HTTPException(status_code=404, detail="Чат не найден")

        user_message, assistant_message = await self.create_dialog_turn(
            db=db,
            chat_id=chat_id,
            user_id=chat.user_id,
            user_text=payload.content,
        )
        return [
            MessageResponse.model_validate(user_message),
            MessageResponse.model_validate(assistant_message),
        ]

    async def create_system_error_response(
        self,
        db: AsyncSession,
        chat_id: UUID,
        error_text: str,
        current_user: User,
    ) -> MessageResponse:
        chat = await chats_service.get_chat(db, chat_id, current_user)
        if not chat:
            raise HTTPException(status_code=404, detail="Чат не найден")

        system_message = await self.create_system_error(
            db=db,
            chat_id=chat_id,
            user_id=chat.user_id,
            error_text=error_text,
        )
        return MessageResponse.model_validate(system_message)


messages_service = MessagesService()