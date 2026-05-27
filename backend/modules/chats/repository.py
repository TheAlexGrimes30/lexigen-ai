from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.chats import Chat
from backend.modules.chats.interfaces import BaseChatsRepository


class ChatsRepository(BaseChatsRepository):
    """Репозиторий операций с чатами."""

    async def list_by_user(
        self,
        db: AsyncSession,
        user_id: UUID,
    ) -> list[Chat]:
        """Возвращает чаты пользователя с сортировкой по дате создания."""
        stmt = (
            select(Chat)
            .where(Chat.user_id == user_id)
            .order_by(Chat.created_at.desc())
        )

        result = await db.execute(stmt)

        return list(result.scalars().all())

    async def get_by_id_and_user(
        self,
        db: AsyncSession,
        chat_id: UUID,
        user_id: UUID,
    ) -> Chat | None:
        """Возвращает чат по идентификатору и владельцу."""
        stmt = select(Chat).where(
            Chat.id == chat_id,
            Chat.user_id == user_id,
        )

        result = await db.execute(stmt)

        return result.scalar_one_or_none()

    async def add(
        self,
        db: AsyncSession,
        chat: Chat,
    ) -> Chat:
        """Добавляет чат и фиксирует транзакцию."""
        db.add(chat)

        await db.commit()
        await db.refresh(chat)

        return chat

    async def save(
        self,
        db: AsyncSession,
        chat: Chat,
    ) -> Chat:
        """Фиксирует изменения чата и обновляет объект."""
        await db.commit()
        await db.refresh(chat)

        return chat

    async def delete(
        self,
        db: AsyncSession,
        chat: Chat,
    ) -> None:
        """Удаляет чат и фиксирует транзакцию."""
        await db.delete(chat)
        await db.commit()
