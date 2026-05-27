from abc import ABC, abstractmethod
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.users import User


class BasePasswordHasher(ABC):
    """Абстракция хеширования и проверки паролей."""

    @abstractmethod
    def hash_password(
        self,
        password: str,
    ) -> str:
        """Возвращает хеш пароля."""
        
        raise NotImplementedError

    @abstractmethod
    def verify_password(
        self,
        raw_password: str,
        hashed_password: str,
    ) -> bool:
        """Проверяет соответствие пароля его хешу."""

        raise NotImplementedError


class BaseTokenManager(ABC):
    """Абстракция создания и декодирования JWT-токенов."""

    @abstractmethod
    def create_access_token(
        self,
        user: User,
    ) -> str:
        """Создаёт JWT access token для пользователя."""

        raise NotImplementedError

    @abstractmethod
    def decode_access_token(
        self,
        token: str,
    ) -> dict:
        """Декодирует и валидирует JWT access token."""

        raise NotImplementedError


class BaseAuthRepository(ABC):
    """Абстрактный репозиторий пользователей и auth-аналитики."""

    @abstractmethod
    async def get_user_by_email(
        self,
        db: AsyncSession,
        email: str,
    ) -> User | None:
        """Возвращает пользователя по email."""

        raise NotImplementedError

    @abstractmethod
    async def get_user_by_id(
        self,
        db: AsyncSession,
        user_id: UUID,
    ) -> User | None:
        """Возвращает пользователя по идентификатору."""

        raise NotImplementedError

    @abstractmethod
    async def add_user(
        self,
        db: AsyncSession,
        user: User,
    ) -> User:
        """Добавляет пользователя в базу данных."""

        raise NotImplementedError

    @abstractmethod
    async def save_user(
        self,
        db: AsyncSession,
        user: User,
    ) -> User:
        """Сохраняет изменения пользователя в базе данных."""

        raise NotImplementedError

    @abstractmethod
    async def count_users(
        self,
        db: AsyncSession,
    ) -> int:
        """Возвращает количество пользователей."""

        raise NotImplementedError

    @abstractmethod
    async def count_chats(
        self,
        db: AsyncSession,
    ) -> int:
        """Возвращает количество чатов."""

        raise NotImplementedError

    @abstractmethod
    async def count_messages(
        self,
        db: AsyncSession,
    ) -> int:
        """Возвращает количество сообщений."""

        raise NotImplementedError


class BaseAuthService(ABC):
    """Абстрактный сервис авторизации."""

    @abstractmethod
    async def register(
        self,
        db: AsyncSession,
        name: str,
        email: str,
        password: str,
    ) -> User:
        """Регистрирует нового пользователя."""

        raise NotImplementedError

    @abstractmethod
    async def login(
        self,
        db: AsyncSession,
        email: str,
        password: str,
    ) -> User:
        """Авторизует пользователя по email и паролю."""

        raise NotImplementedError
