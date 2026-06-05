from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.logger_config import get_logger
from backend.db.chat_documents import ChatDocument
from backend.db.messages import Message
from backend.modules.messages.interfaces import BaseMessagesRepository

logger = get_logger(__name__)


class MessagesRepository(BaseMessagesRepository):
    """Репозиторий сообщений, документов и результатов анализа."""

    async def list_by_chat(
        self,
        db: AsyncSession,
        chat_id: UUID,
    ) -> list[Message]:
        """Возвращает сообщения чата по возрастанию даты создания."""

        logger.info(
            "Fetching messages by chat: chat_id=%s",
            chat_id,
        )

        try:
            stmt = (
                select(Message)
                .where(Message.chat_id == chat_id)
                .order_by(Message.created_at.asc())
            )

            result = await db.execute(stmt)
            messages = list(result.scalars().all())

            logger.info(
                "Messages fetched successfully: chat_id=%s, count=%s",
                chat_id,
                len(messages),
            )

            return messages

        except Exception:
            logger.exception(
                "Failed to fetch messages by chat: chat_id=%s",
                chat_id,
            )
            raise

    async def count_user_documents(
        self,
        db: AsyncSession,
        user_id: UUID,
    ) -> int:
        """Возвращает количество загруженных документов пользователя."""

        logger.info(
            "Counting uploaded documents: user_id=%s",
            user_id,
        )

        try:
            value = await db.scalar(
                select(func.count(ChatDocument.id)).where(
                    ChatDocument.uploaded_by == user_id,
                )
            )

            count = int(value or 0)

            logger.info(
                "Uploaded documents counted: user_id=%s, count=%s",
                user_id,
                count,
            )

            return count

        except Exception:
            logger.exception(
                "Failed to count uploaded documents: user_id=%s",
                user_id,
            )
            raise

    async def add_and_flush(
        self,
        db: AsyncSession,
        entity: object,
    ) -> object:
        """Добавляет объект в сессию и выполняет flush."""

        entity_type = type(entity).__name__

        logger.info(
            "Adding entity and flushing session: entity_type=%s",
            entity_type,
        )

        try:
            db.add(entity)
            await db.flush()

            logger.info(
                "Entity added and flushed successfully: entity_type=%s",
                entity_type,
            )

            return entity

        except Exception:
            logger.exception(
                "Failed to add and flush entity: entity_type=%s",
                entity_type,
            )
            raise

    async def commit_and_refresh_many(
        self,
        db: AsyncSession,
        entities: list[object],
    ) -> None:
        """Фиксирует транзакцию и обновляет переданные объекты."""

        logger.info(
            "Committing and refreshing entities: count=%s",
            len(entities),
        )

        try:
            await db.commit()

            for entity in entities:
                await db.refresh(entity)

            logger.info(
                "Entities committed and refreshed successfully: count=%s",
                len(entities),
            )

        except Exception:
            logger.exception(
                "Failed to commit and refresh entities: count=%s",
                len(entities),
            )
            raise
