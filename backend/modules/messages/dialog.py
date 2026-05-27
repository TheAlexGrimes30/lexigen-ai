from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.enums import MessageRole
from backend.db.messages import Message
from backend.modules.messages.interfaces import BaseMessagesRepository
from backend.modules.rag.service import rag_app_service


class DialogTurnHandler:
    """Обработчик обычного текстового диалога с RAG."""

    def __init__(
        self,
        repository: BaseMessagesRepository,
    ) -> None:
        """Инициализирует обработчик диалоговых сообщений."""
        self.repository = repository

    async def create_turn(
        self,
        db: AsyncSession,
        chat_id: UUID,
        user_id: UUID,
        user_text: str,
    ) -> tuple[Message, Message]:
        """Создаёт пользовательское сообщение и ответ RAG."""
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

        return user_message, assistant_message

    async def _ask_rag(
        self,
        user_text: str,
    ) -> tuple[str, MessageRole]:
        """Получает ответ RAG или системную ошибку."""
        try:
            assistant_text = await rag_app_service.ask(user_text)

            return assistant_text, MessageRole.assistant
        except Exception as exc:
            return (
                f"Система не смогла получить RAG-ответ: {exc}",
                MessageRole.system,
            )
