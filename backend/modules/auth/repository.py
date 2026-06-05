from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.logger_config import get_logger
from backend.db import User, Message, Chat
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

    async def add_user(
        self,
        db: AsyncSession,
        user: User,
    ) -> User:
        """Добавляет пользователя и фиксирует транзакцию."""

        logger.info(
            "Adding user: email=%s, role=%s",
            user.email,
            user.role.value,
        )

        try:
            db.add(user)

            await db.commit()
            await db.refresh(user)

            logger.info(
                "User added successfully: user_id=%s, email=%s",
                user.id,
                user.email,
            )

            return user

        except Exception:
            logger.exception(
                "Failed to add user: email=%s",
                user.email,
            )
            raise

    async def save_user(
        self,
        db: AsyncSession,
        user: User,
    ) -> User:
        """Фиксирует изменения пользователя и обновляет объект."""

        logger.info(
            "Saving user: user_id=%s",
            user.id,
        )

        try:
            await db.commit()
            await db.refresh(user)

            logger.info(
                "User saved successfully: user_id=%s",
                user.id,
            )

            return user

        except Exception:
            logger.exception(
                "Failed to save user: user_id=%s",
                user.id,
            )
            raise

    async def count_users(
        self,
        db: AsyncSession,
    ) -> int:
        """Возвращает количество пользователей."""

        logger.info("Counting users")

        try:
            value = (
                await db.execute(
                    select(func.count()).select_from(User)
                )
            ).scalar_one()

            result = int(value or 0)

            logger.info(
                "Users counted: count=%s",
                result,
            )

            return result

        except Exception:
            logger.exception("Failed to count users")
            raise

    async def count_chats(
        self,
        db: AsyncSession,
    ) -> int:
        """Возвращает количество чатов."""

        logger.info("Counting chats")

        try:
            value = (
                await db.execute(
                    select(func.count()).select_from(Chat)
                )
            ).scalar_one()

            result = int(value or 0)

            logger.info(
                "Chats counted: count=%s",
                result,
            )

            return result

        except Exception:
            logger.exception("Failed to count chats")
            raise

    async def count_messages(
        self,
        db: AsyncSession,
    ) -> int:
        """Возвращает количество сообщений."""

        logger.info("Counting messages")

        try:
            value = (
                await db.execute(
                    select(func.count()).select_from(Message)
                )
            ).scalar_one()

            result = int(value or 0)

            logger.info(
                "Messages counted: count=%s",
                result,
            )

            return result

        except Exception:
            logger.exception("Failed to count messages")
            raise