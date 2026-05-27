from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import Subscription
from backend.modules.subscriptions.interfaces import BaseSubscriptionsRepository


class SubscriptionsRepository(BaseSubscriptionsRepository):
    """Репозиторий операций с подписками."""

    async def get_active_by_user_id(self, db: AsyncSession, user_id: UUID) -> Subscription | None:
        """Возвращает активную подписку пользователя."""
        stmt = select(Subscription).where(
            Subscription.user_id == user_id,
            Subscription.is_active.is_(True),
        )

        result = await db.execute(stmt)

        return result.scalar_one_or_none()

    async def delete_by_user_id(self, db: AsyncSession, user_id: UUID) -> None:
        """Физически удаляет все подписки пользователя."""
        await db.execute(
            delete(Subscription).where(
                Subscription.user_id == user_id,
            )
        )

    async def add(self, db: AsyncSession, subscription: Subscription) -> Subscription:
        """Добавляет подписку в текущую сессию."""
        db.add(subscription)

        return subscription

    async def commit(self, db: AsyncSession) -> None:
        """Фиксирует текущую транзакцию."""
        await db.commit()
