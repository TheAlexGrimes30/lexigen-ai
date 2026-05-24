from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.enums import MessageRole
from backend.db.messages import Message


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

        assistant_message = Message(
            chat_id=chat_id,
            user_id=user_id,
            role=MessageRole.assistant,
            content=f"Принял ваш запрос по кредитному договору: {user_text}",
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


messages_service = MessagesService()
