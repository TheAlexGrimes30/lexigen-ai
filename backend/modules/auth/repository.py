from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.logger_config import get_logger
from backend.db import User
from backend.modules.auth.interfaces import BaseAuthRepository

logger = get_logger(__name__)


class AuthRepository(BaseAuthRepository):
    """Репозиторий пользователей и статистики auth-модуля."""

    async def get_user_by_email(
        self,
        db: AsyncSession,
        email: str,
    ) -> User | None:
        """Возвращает пользователя по нормализованному email."""

        normalized_email = email.lower()

        logger.info(
            "Fetching user by email: email=%s",
            normalized_email,
        )

        try:
            stmt = select(User).where(
                User.email == normalized_email
            )

            result = await db.execute(stmt)
            user = result.scalar_one_or_none()

            logger.info(
                "User by email fetched: email=%s, found=%s",
                normalized_email,
                user is not None,
            )

            return user

        except Exception:
            logger.exception(
                "Failed to fetch user by email: email=%s",
                normalized_email,
            )
            raise

    async def get_user_by_id(
        self,
        db: AsyncSession,
        user_id: UUID,
    ) -> User | None:
        """Возвращает пользователя по UUID."""

        logger.info(
            "Fetching user by id: user_id=%s",
            user_id,
        )

        try:
            stmt = select(User).where(
                User.id == user_id
            )

            result = await db.execute(stmt)
            user = result.scalar_one_or_none()

            logger.info(
                "User by id fetched: user_id=%s, found=%s",
                user_id,
                user is not None,
            )

            return user

        except Exception:
            logger.exception(
                "Failed to fetch user by id: user_id=%s",
                user_id,
            )
            raise

