from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.logger_config import get_logger
from backend.db.chats import Chat
from backend.db.users import User
from backend.modules.chats.interfaces import (
    BaseChatsRepository,
    BaseChatsService,
)
from backend.modules.chats.mapper import ChatMapper
from backend.modules.chats.repository import ChatsRepository
from backend.modules.chats.schema import (
    ChatCreateRequest,
    ChatResponse,
    ChatUpdateRequest,
)


logger = get_logger(__name__)


class ChatsService(BaseChatsService):
    """Сервис бизнес-логики чатов."""

    def __init__(
        self,
        repository: BaseChatsRepository,
        mapper: ChatMapper,
    ) -> None:
        """Инициализирует сервис чатов."""
        self.repository = repository
        self.mapper = mapper

    async def list_chats(
        self,
        db: AsyncSession,
        current_user: User,
    ) -> list[Chat]:
        """Возвращает список чатов текущего пользователя."""

        logger.info(
            "Listing chats: user_id=%s",
            current_user.id,
        )

        return await self.repository.list_by_user(
            db=db,
            user_id=current_user.id,
        )

    async def create_chat(
        self,
        db: AsyncSession,
        title: str,
        current_user: User,
    ) -> Chat:
        """Создаёт новый чат для текущего пользователя."""

        logger.info(
            "Creating chat: user_id=%s, title=%s",
            current_user.id,
            title,
        )

        chat = Chat(
            title=title,
            user_id=current_user.id,
        )

        created_chat = await self.repository.add(
            db=db,
            chat=chat,
        )

        logger.info(
            "Chat created: chat_id=%s",
            created_chat.id,
        )

        return created_chat

    async def get_chat(
        self,
        db: AsyncSession,
        chat_id: UUID,
        current_user: User,
    ) -> Chat | None:
        """Возвращает чат текущего пользователя по идентификатору."""

        logger.info(
            "Getting chat: chat_id=%s, user_id=%s",
            chat_id,
            current_user.id,
        )

        return await self.repository.get_by_id_and_user(
            db=db,
            chat_id=chat_id,
            user_id=current_user.id,
        )

    async def update_chat(
        self,
        db: AsyncSession,
        chat_id: UUID,
        title: str,
        current_user: User,
    ) -> Chat | None:
        """Изменяет название чата текущего пользователя."""

        logger.info(
            "Updating chat: chat_id=%s, user_id=%s",
            chat_id,
            current_user.id,
        )

        chat = await self.get_chat(
            db=db,
            chat_id=chat_id,
            current_user=current_user,
        )

        if not chat:
            logger.warning(
                "Chat not found for update: chat_id=%s, user_id=%s",
                chat_id,
                current_user.id,
            )

            return None

        chat.title = title.strip()

        updated_chat = await self.repository.save(
            db=db,
            chat=chat,
        )

        logger.info(
            "Chat updated successfully: chat_id=%s",
            updated_chat.id,
        )

        return updated_chat

    async def delete_chat(
        self,
        db: AsyncSession,
        chat_id: UUID,
        current_user: User,
    ) -> bool:
        """Удаляет чат текущего пользователя."""

        logger.info(
            "Deleting chat: chat_id=%s, user_id=%s",
            chat_id,
            current_user.id,
        )

        chat = await self.get_chat(
            db=db,
            chat_id=chat_id,
            current_user=current_user,
        )

        if not chat:
            logger.warning(
                "Chat not found for deletion: chat_id=%s, user_id=%s",
                chat_id,
                current_user.id,
            )

            return False

        await self.repository.delete(
            db=db,
            chat=chat,
        )

        logger.info(
            "Chat deleted successfully: chat_id=%s",
            chat_id,
        )

        return True

    async def list_chats_response(
        self,
        db: AsyncSession,
        current_user: User,
    ) -> list[ChatResponse]:
        """Возвращает DTO списка чатов текущего пользователя."""

        chats = await self.list_chats(
            db=db,
            current_user=current_user,
        )

        return self.mapper.to_response_list(chats)

    async def create_chat_response(
        self,
        db: AsyncSession,
        payload: ChatCreateRequest,
        current_user: User,
    ) -> ChatResponse:
        """Создаёт чат и возвращает DTO ответа."""

        chat = await self.create_chat(
            db=db,
            title=payload.title,
            current_user=current_user,
        )

        return self.mapper.to_response(chat)

    async def update_chat_response(
        self,
        db: AsyncSession,
        chat_id: UUID,
        payload: ChatUpdateRequest,
        current_user: User,
    ) -> ChatResponse | None:
        """Изменяет чат и возвращает DTO ответа."""

        chat = await self.update_chat(
            db=db,
            chat_id=chat_id,
            title=payload.title,
            current_user=current_user,
        )

        if not chat:
            return None

        return self.mapper.to_response(chat)

    async def delete_chat_response(
        self,
        db: AsyncSession,
        chat_id: UUID,
        current_user: User,
    ) -> dict[str, str]:
        """Удаляет чат и возвращает статус удаления."""

        deleted = await self.delete_chat(
            db=db,
            chat_id=chat_id,
            current_user=current_user,
        )

        if not deleted:
            return {"status": "not_found"}

        return {"status": "deleted"}


chats_service = ChatsService(
    repository=ChatsRepository(),
    mapper=ChatMapper(),
)