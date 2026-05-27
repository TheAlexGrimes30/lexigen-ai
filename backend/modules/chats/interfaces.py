from abc import ABC, abstractmethod
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.chats import Chat
from backend.db.users import User
from backend.modules.chats.schema import (
    ChatCreateRequest,
    ChatResponse,
    ChatUpdateRequest,
)


class BaseChatsRepository(ABC):
    """Абстрактный репозиторий чатов."""

    @abstractmethod
    async def list_by_user(
        self,
        db: AsyncSession,
        user_id: UUID,
    ) -> list[Chat]:
        """Возвращает список чатов пользователя."""
        raise NotImplementedError

    @abstractmethod
    async def get_by_id_and_user(
        self,
        db: AsyncSession,
        chat_id: UUID,
        user_id: UUID,
    ) -> Chat | None:
        """Возвращает чат пользователя по идентификатору."""
        raise NotImplementedError

    @abstractmethod
    async def add(
        self,
        db: AsyncSession,
        chat: Chat,
    ) -> Chat:
        """Добавляет чат в базу данных."""
        raise NotImplementedError

    @abstractmethod
    async def save(
        self,
        db: AsyncSession,
        chat: Chat,
    ) -> Chat:
        """Сохраняет изменения чата в базе данных."""
        raise NotImplementedError

    @abstractmethod
    async def delete(
        self,
        db: AsyncSession,
        chat: Chat,
    ) -> None:
        """Удаляет чат из базы данных."""
        raise NotImplementedError


class BaseChatsService(ABC):
    """Абстрактный сервис чатов."""

    @abstractmethod
    async def list_chats(
        self,
        db: AsyncSession,
        current_user: User,
    ) -> list[Chat]:
        """Возвращает список чатов текущего пользователя."""
        raise NotImplementedError

    @abstractmethod
    async def create_chat(
        self,
        db: AsyncSession,
        title: str,
        current_user: User,
    ) -> Chat:
        """Создаёт чат для текущего пользователя."""
        raise NotImplementedError

    @abstractmethod
    async def get_chat(
        self,
        db: AsyncSession,
        chat_id: UUID,
        current_user: User,
    ) -> Chat | None:
        """Возвращает чат текущего пользователя."""
        raise NotImplementedError

    @abstractmethod
    async def update_chat(
        self,
        db: AsyncSession,
        chat_id: UUID,
        title: str,
        current_user: User,
    ) -> Chat | None:
        """Изменяет название чата текущего пользователя."""
        raise NotImplementedError

    @abstractmethod
    async def delete_chat(
        self,
        db: AsyncSession,
        chat_id: UUID,
        current_user: User,
    ) -> bool:
        """Удаляет чат текущего пользователя."""
        raise NotImplementedError

    @abstractmethod
    async def list_chats_response(
        self,
        db: AsyncSession,
        current_user: User,
    ) -> list[ChatResponse]:
        """Возвращает DTO списка чатов текущего пользователя."""
        raise NotImplementedError

    @abstractmethod
    async def create_chat_response(
        self,
        db: AsyncSession,
        payload: ChatCreateRequest,
        current_user: User,
    ) -> ChatResponse:
        """Создаёт чат и возвращает DTO."""
        raise NotImplementedError

    @abstractmethod
    async def update_chat_response(
        self,
        db: AsyncSession,
        chat_id: UUID,
        payload: ChatUpdateRequest,
        current_user: User,
    ) -> ChatResponse | None:
        """Изменяет чат и возвращает DTO."""
        raise NotImplementedError

    @abstractmethod
    async def delete_chat_response(
        self,
        db: AsyncSession,
        chat_id: UUID,
        current_user: User,
    ) -> dict[str, str]:
        """Удаляет чат и возвращает статус операции."""
        raise NotImplementedError
