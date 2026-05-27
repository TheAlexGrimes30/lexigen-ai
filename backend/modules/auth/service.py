from datetime import datetime, timezone
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import settings
from backend.db.enums import UserRole
from backend.db.users import User
from backend.modules.auth.interfaces import (
    BaseAuthRepository,
    BaseAuthService,
    BasePasswordHasher,
    BaseTokenManager,
)
from backend.modules.auth.repository import AuthRepository
from backend.modules.auth.schema import AdminAnalyticsResponse, AuthUserResponse
from backend.modules.auth.security import JwtTokenManager, Sha256PasswordHasher


class AuthService(BaseAuthService):
    """Сервис авторизации и управления пользователями."""

    def __init__(
        self,
        repository: BaseAuthRepository,
        password_hasher: BasePasswordHasher,
        token_manager: BaseTokenManager,
    ) -> None:
        """Инициализирует сервис авторизации."""
        self.repository = repository
        self.password_hasher = password_hasher
        self.token_manager = token_manager

    def hash_password(
        self,
        password: str,
    ) -> str:
        """Возвращает хеш пароля."""
        return self.password_hasher.hash_password(password)

    def verify_password(
        self,
        raw_password: str,
        hashed_password: str,
    ) -> bool:
        """Проверяет пароль пользователя."""
        return self.password_hasher.verify_password(
            raw_password,
            hashed_password,
        )

    def create_access_token(
        self,
        user: User,
    ) -> str:
        """Создаёт JWT access token для пользователя."""
        return self.token_manager.create_access_token(user)

    def decode_access_token(
        self,
        token: str,
    ) -> dict:
        """Декодирует JWT access token."""
        return self.token_manager.decode_access_token(token)

    async def get_user_by_email(
        self,
        db: AsyncSession,
        email: str,
    ) -> User | None:
        """Возвращает пользователя по email."""
        return await self.repository.get_user_by_email(
            db,
            email,
        )

    async def get_user_by_id(
        self,
        db: AsyncSession,
        user_id: UUID,
    ) -> User | None:
        """Возвращает пользователя по идентификатору."""
        return await self.repository.get_user_by_id(
            db,
            user_id,
        )

    async def register(
        self,
        db: AsyncSession,
        name: str,
        email: str,
        password: str,
    ) -> User:
        """Регистрирует пользователя с ролью user или admin."""
        normalized_email = email.lower().strip()

        existing = await self.get_user_by_email(
            db,
            normalized_email,
        )

        if existing:
            raise HTTPException(
                status_code=409,
                detail="Пользователь с таким email уже существует",
            )

        user = User(
            name=name.strip(),
            email=normalized_email,
            password_hash=self.hash_password(password),
            role=(
                UserRole.admin
                if normalized_email == settings.ADMIN_EMAIL.lower()
                else UserRole.user
            ),
        )

        return await self.repository.add_user(
            db,
            user,
        )

    async def login(
        self,
        db: AsyncSession,
        email: str,
        password: str,
    ) -> User:
        """Авторизует пользователя и обновляет дату последнего входа."""
        normalized_email = email.lower().strip()

        user = await self.get_user_by_email(
            db,
            normalized_email,
        )

        if not user or not self.verify_password(
            password,
            user.password_hash,
        ):
            raise HTTPException(
                status_code=401,
                detail="Неверный email или пароль",
            )

        user.last_login_at = datetime.now(timezone.utc)

        return await self.repository.save_user(
            db,
            user,
        )

    async def promote_to_admin(
        self,
        db: AsyncSession,
        user: User,
    ) -> User:
        """Повышает пользователя до администратора."""
        if user.role != UserRole.admin:
            user.role = UserRole.admin

            return await self.repository.save_user(
                db,
                user,
            )

        return user

    async def get_admin_analytics(
        self,
        db: AsyncSession,
    ) -> AdminAnalyticsResponse:
        """Возвращает базовую административную аналитику."""
        users_count = await self.repository.count_users(db)
        chats_count = await self.repository.count_chats(db)
        messages_count = await self.repository.count_messages(db)

        return AdminAnalyticsResponse(
            users_count=users_count,
            chats_count=chats_count,
            messages_count=messages_count,
        )

    def to_auth_user(
        self,
        user: User,
    ) -> AuthUserResponse:
        """Преобразует ORM-пользователя в DTO ответа."""
        return AuthUserResponse(
            id=user.id,
            name=user.name,
            email=user.email,
            role=user.role.value,
            created_at=user.created_at,
        )


auth_service = AuthService(
    repository=AuthRepository(),
    password_hasher=Sha256PasswordHasher(),
    token_manager=JwtTokenManager(),
)
