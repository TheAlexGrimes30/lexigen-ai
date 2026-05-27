from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.chat_documents import ChatDocument
from backend.db.messages import Message
from backend.modules.messages.interfaces import BaseMessagesRepository


class MessagesRepository(BaseMessagesRepository):
    """Репозиторий сообщений, документов и результатов анализа."""

    async def list_by_chat(
        self,
        db: AsyncSession,
        chat_id: UUID,
    ) -> list[Message]:
        """Возвращает сообщения чата по возрастанию даты создания."""
        stmt = (
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.created_at.asc())
        )

        result = await db.execute(stmt)

        return list(result.scalars().all())

    async def count_user_documents(
        self,
        db: AsyncSession,
        user_id: UUID,
    ) -> int:
        """Возвращает количество загруженных документов пользователя."""
        value = await db.scalar(
            select(func.count(ChatDocument.id)).where(
                ChatDocument.uploaded_by == user_id,
            )
        )

        return int(value or 0)

    async def add_and_flush(
        self,
        db: AsyncSession,
        entity: object,
    ) -> object:
        """Добавляет объект в сессию и выполняет flush."""
        db.add(entity)

        await db.flush()

        return entity

    async def commit_and_refresh_many(
        self,
        db: AsyncSession,
        entities: list[object],
    ) -> None:
        """Фиксирует транзакцию и обновляет переданные объекты."""
        await db.commit()

        for entity in entities:
            await db.refresh(entity)
