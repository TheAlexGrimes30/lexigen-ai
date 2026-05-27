from abc import ABC, abstractmethod
from uuid import UUID

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.chat_documents import ChatDocument
from backend.db.messages import Message
from backend.db.users import User
from backend.modules.messages.schema import MessageResponse


class BaseMessagesRepository(ABC):
    """Абстрактный репозиторий сообщений и документов чата."""

    @abstractmethod
    async def list_by_chat(
        self,
        db: AsyncSession,
        chat_id: UUID,
    ) -> list[Message]:
        """Возвращает сообщения чата."""
        raise NotImplementedError

    @abstractmethod
    async def count_user_documents(
        self,
        db: AsyncSession,
        user_id: UUID,
    ) -> int:
        """Возвращает количество документов пользователя."""
        raise NotImplementedError

    @abstractmethod
    async def add_and_flush(
        self,
        db: AsyncSession,
        entity: object,
    ) -> object:
        """Добавляет объект в сессию и выполняет flush."""
        raise NotImplementedError

    @abstractmethod
    async def commit_and_refresh_many(
        self,
        db: AsyncSession,
        entities: list[object],
    ) -> None:
        """Фиксирует транзакцию и обновляет список объектов."""
        raise NotImplementedError


class BaseDocumentParser(ABC):
    """Абстракция парсера загруженных документов."""

    @abstractmethod
    async def extract_text_from_upload(
        self,
        file: UploadFile,
    ) -> str:
        """Извлекает текст из загруженного документа."""
        raise NotImplementedError


class BaseDocumentAnalysisPolicy(ABC):
    """Абстракция политики доступа к анализу документов."""

    @abstractmethod
    async def ensure_allowed(
        self,
        db: AsyncSession,
        current_user: User,
    ) -> None:
        """Проверяет право пользователя на анализ документа."""
        raise NotImplementedError


class BaseMessagesService(ABC):
    """Абстрактный сервис сообщений."""

    @abstractmethod
    async def list_messages_response(
        self,
        db: AsyncSession,
        chat_id: UUID,
        current_user: User,
    ) -> list[MessageResponse]:
        """Возвращает DTO списка сообщений чата."""
        raise NotImplementedError

    @abstractmethod
    async def create_message_turn_response(
        self,
        db: AsyncSession,
        chat_id: UUID,
        content: str,
        file: UploadFile | None,
        current_user: User,
    ) -> list[MessageResponse]:
        """Создаёт пользовательское и ответное сообщение."""
        raise NotImplementedError
