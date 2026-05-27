from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import Chat, User, Message
from backend.modules.auth.interfaces import BaseAuthRepository


class AuthRepository(BaseAuthRepository):
    """Репозиторий пользователей и статистики auth-модуля."""

    async def get_user_by_email(
        self,
        db: AsyncSession,
        email: str,
    ) -> User | None:
        """Возвращает пользователя по нормализованному email."""
        stmt = select(User).where(
            User.email == email.lower()
        )

        result = await db.execute(stmt)

        return result.scalar_one_or_none()

    async def get_user_by_id(
        self,
        db: AsyncSession,
        user_id: UUID,
    ) -> User | None:
        """Возвращает пользователя по UUID."""
        stmt = select(User).where(
            User.id == user_id
        )

        result = await db.execute(stmt)

        return result.scalar_one_or_none()

    async def add_user(
        self,
        db: AsyncSession,
        user: User,
    ) -> User:
        """Добавляет пользователя и фиксирует транзакцию."""
        db.add(user)

        await db.commit()
        await db.refresh(user)

        return user

    async def save_user(
        self,
        db: AsyncSession,
        user: User,
    ) -> User:
        """Фиксирует изменения пользователя и обновляет объект."""
        await db.commit()
        await db.refresh(user)

        return user

    async def count_users(
        self,
        db: AsyncSession,
    ) -> int:
        """Возвращает количество пользователей."""
        value = (
            await db.execute(
                select(func.count()).select_from(User)
            )
        ).scalar_one()

        return int(value or 0)

    async def count_chats(
        self,
        db: AsyncSession,
    ) -> int:
        """Возвращает количество чатов."""

        value = (
            await db.execute(
                select(func.count()).select_from(Chat)
            )
        ).scalar_one()

        return int(value or 0)

    async def count_messages(
        self,
        db: AsyncSession,
    ) -> int:
        """Возвращает количество сообщений."""

        value = (
            await db.execute(
                select(func.count()).select_from(Message)
            )
        ).scalar_one()

        return int(value or 0)
