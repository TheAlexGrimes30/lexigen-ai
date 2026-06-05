from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.logger_config import get_logger
from backend.db.enums import MessageRole
from backend.db.messages import Message
from backend.modules.messages.interfaces import BaseMessagesRepository
from backend.modules.rag.service import rag_app_service

logger = get_logger(__name__)


class DialogTurnHandler:
    """Обработчик обычного текстового диалога с RAG."""

    def __init__(
        self,
        repository: BaseMessagesRepository,
    ):
        self.repository = repository

    async def create_turn(
        self,
        db: AsyncSession,
        chat_id: UUID,
        user_id: UUID,
        user_text: str,
    ) -> tuple[Message, Message]:
        """Создаёт пользовательское сообщение и ответ RAG."""

        logger.info(
            "Dialog turn creation started: chat_id=%s, user_id=%s",
            chat_id,
            user_id,
        )

        try:
            user_message = Message(
                chat_id=chat_id,
                user_id=user_id,
                role=MessageRole.user,
                content=user_text,
            )

            await self.repository.add_and_flush(db, user_message)

            assistant_text, assistant_role = await self._ask_rag(user_text)

            assistant_message = Message(
                chat_id=chat_id,
                user_id=user_id,
                role=assistant_role,
                content=assistant_text,
            )

            await self.repository.add_and_flush(db, assistant_message)

            await self.repository.commit_and_refresh_many(
                db=db,
                entities=[
                    user_message,
                    assistant_message,
                ],
            )

            logger.info(
                "Dialog turn created successfully: chat_id=%s, user_id=%s, assistant_role=%s",
                chat_id,
                user_id,
                assistant_role.value,
            )

            return user_message, assistant_message

        except Exception:
            logger.exception(
                "Failed to create dialog turn: chat_id=%s, user_id=%s",
                chat_id,
                user_id,
            )
            raise

    async def _ask_rag(
        self,
        user_text: str,
    ) -> tuple[str, MessageRole]:
        """Получает ответ RAG или системную ошибку."""

        logger.info("RAG dialog request started")

        try:
            assistant_text = await rag_app_service.ask(user_text)

            logger.info("RAG dialog request completed successfully")

            return assistant_text, MessageRole.assistant

        except Exception as exc:
            logger.exception("RAG dialog request failed")

            return (
                f"Система не смогла получить RAG-ответ: {exc}",
                MessageRole.system,
            )
