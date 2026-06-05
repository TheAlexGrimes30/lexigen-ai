from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.logger_config import get_logger
from backend.db.chats import Chat
from backend.modules.chats.interfaces import BaseChatsRepository

logger = get_logger(__name__)


class ChatsRepository(BaseChatsRepository):
    """Репозиторий операций с чатами."""

    async def list_by_user(
        self,
        db: AsyncSession,
        user_id: UUID,
    ) -> list[Chat]:
        """Возвращает чаты пользователя с сортировкой по дате создания."""

        logger.info(
            "Fetching chats by user: user_id=%s",
            user_id,
        )

        try:
            stmt = (
                select(Chat)
                .where(Chat.user_id == user_id)
                .order_by(Chat.created_at.desc())
            )

            result = await db.execute(stmt)
            chats = list(result.scalars().all())

            logger.info(
                "Chats fetched successfully: user_id=%s, count=%s",
                user_id,
                len(chats),
            )

            return chats

        except Exception:
            logger.exception(
                "Failed to fetch chats: user_id=%s",
                user_id,
            )
            raise

    async def get_by_id_and_user(
        self,
        db: AsyncSession,
        chat_id: UUID,
        user_id: UUID,
    ) -> Chat | None:
        """Возвращает чат по идентификатору и владельцу."""

        logger.info(
            "Fetching chat: chat_id=%s, user_id=%s",
            chat_id,
            user_id,
        )

        try:
            stmt = select(Chat).where(
                Chat.id == chat_id,
                Chat.user_id == user_id,
            )

            result = await db.execute(stmt)
            chat = result.scalar_one_or_none()

            logger.info(
                "Chat fetched: chat_id=%s, found=%s",
                chat_id,
                chat is not None,
            )

            return chat

        except Exception:
            logger.exception(
                "Failed to fetch chat: chat_id=%s, user_id=%s",
                chat_id,
                user_id,
            )
            raise

    async def add(
        self,
        db: AsyncSession,
        chat: Chat,
    ) -> Chat:
        """Добавляет чат и фиксирует транзакцию."""

        logger.info(
            "Creating chat: user_id=%s, title=%s",
            chat.user_id,
            chat.title,
        )

        try:
            db.add(chat)

            await db.commit()
            await db.refresh(chat)

            logger.info(
                "Chat created successfully: chat_id=%s, user_id=%s",
                chat.id,
                chat.user_id,
            )

            return chat

        except Exception:
            logger.exception(
                "Failed to create chat: user_id=%s",
                chat.user_id,
            )
            raise

    async def save(
        self,
        db: AsyncSession,
        chat: Chat,
    ) -> Chat:
        """Фиксирует изменения чата и обновляет объект."""

        logger.info(
            "Updating chat: chat_id=%s",
            chat.id,
        )

        try:
            await db.commit()
            await db.refresh(chat)

            logger.info(
                "Chat updated successfully: chat_id=%s",
                chat.id,
            )

            return chat

        except Exception:
            logger.exception(
                "Failed to update chat: chat_id=%s",
                chat.id,
            )
            raise

    async def delete(
        self,
        db: AsyncSession,
        chat: Chat,
    ) -> None:
        """Удаляет чат и фиксирует транзакцию."""

        logger.info(
            "Deleting chat: chat_id=%s",
            chat.id,
        )

        try:
            await db.delete(chat)
            await db.commit()

            logger.info(
                "Chat deleted successfully: chat_id=%s",
                chat.id,
            )

        except Exception:
            logger.exception(
                "Failed to delete chat: chat_id=%s",
                chat.id,
            )
            raise