from datetime import datetime, timezone
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import settings
from backend.app.logger_config import get_logger
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

logger = get_logger(__name__)


class AuthService(BaseAuthService):
    """Сервис авторизации и управления пользователями."""

    def __init__(
        self,
        repository: BaseAuthRepository,
        password_hasher: BasePasswordHasher,
        token_manager: BaseTokenManager,
    ):
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

        logger.info(
            "Creating access token: user_id=%s",
            user.id,
        )

        return self.token_manager.create_access_token(user)

    def decode_access_token(
        self,
        token: str,
    ) -> dict:
        """Декодирует JWT access token."""

        logger.info("Decoding access token")

        try:
            payload = self.token_manager.decode_access_token(token)

            logger.info("Access token decoded successfully")

            return payload

        except Exception:
            logger.warning("Access token decoding failed")
            raise

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

        logger.info(
            "User registration started: email=%s",
            normalized_email,
        )

        existing = await self.get_user_by_email(
            db,
            normalized_email,
        )

        if existing:
            logger.warning(
                "User registration rejected: email already exists: email=%s",
                normalized_email,
            )

            raise HTTPException(
                status_code=409,
                detail="Пользователь с таким email уже существует",
            )

        role = (
            UserRole.admin
            if normalized_email == settings.ADMIN_EMAIL.lower()
            else UserRole.user
        )

        user = User(
            name=name.strip(),
            email=normalized_email,
            password_hash=self.hash_password(password),
            role=role,
        )

        created_user = await self.repository.add_user(
            db,
            user,
        )

        logger.info(
            "User registered successfully: user_id=%s, email=%s, role=%s",
            created_user.id,
            created_user.email,
            created_user.role.value,
        )

        return created_user

    async def login(
        self,
        db: AsyncSession,
        email: str,
        password: str,
    ) -> User:
        """Авторизует пользователя и обновляет дату последнего входа."""

        normalized_email = email.lower().strip()

        logger.info(
            "User login attempt: email=%s",
            normalized_email,
        )

        user = await self.get_user_by_email(
            db,
            normalized_email,
        )

        if not user or not self.verify_password(
            password,
            user.password_hash,
        ):
            logger.warning(
                "User login failed: invalid credentials: email=%s",
                normalized_email,
            )

            raise HTTPException(
                status_code=401,
                detail="Неверный email или пароль",
            )

        user.last_login_at = datetime.now(timezone.utc)

        saved_user = await self.repository.save_user(
            db,
            user,
        )

        logger.info(
            "User logged in successfully: user_id=%s, email=%s",
            saved_user.id,
            saved_user.email,
        )

        return saved_user

    async def promote_to_admin(
        self,
        db: AsyncSession,
        user: User,
    ) -> User:
        """Повышает пользователя до администратора."""

        logger.info(
            "Admin promotion requested: user_id=%s, current_role=%s",
            user.id,
            user.role.value,
        )

        if user.role != UserRole.admin:
            user.role = UserRole.admin

            promoted_user = await self.repository.save_user(
                db,
                user,
            )

            logger.info(
                "User promoted to admin: user_id=%s",
                promoted_user.id,
            )

            return promoted_user

        logger.info(
            "Admin promotion skipped: user is already admin: user_id=%s",
            user.id,
        )

        return user

    async def get_admin_analytics(
        self,
        db: AsyncSession,
    ) -> AdminAnalyticsResponse:
        """Возвращает базовую административную аналитику."""

        logger.info("Auth admin analytics calculation started")

        users_count = await self.repository.count_users(db)
        chats_count = await self.repository.count_chats(db)
        messages_count = await self.repository.count_messages(db)

        logger.info(
            "Auth admin analytics calculated: users=%s, chats=%s, messages=%s",
            users_count,
            chats_count,
            messages_count,
        )

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