from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import User, Subscription


class AdminAnalyticsRepository:
    """Репозиторий административной аналитики."""

    async def count_users(
        self,
        db: AsyncSession,
    ) -> int:
        """Возвращает общее количество зарегистрированных пользователей."""
        total_users = await db.scalar(
            select(func.count(User.id))
        )

        return int(total_users or 0)

    async def count_users_by_active_subscription(
        self,
        db: AsyncSession,
    ) -> list[tuple[object, int]]:
        """Возвращает количество пользователей по активным подпискам."""

        stmt = (
            select(
                Subscription.plan_name,
                func.count(func.distinct(Subscription.user_id)),
            )
            .where(Subscription.is_active.is_(True))
            .group_by(Subscription.plan_name)
        )

        result = await db.execute(stmt)

        return [
            (
                plan,
                int(count or 0),
            )
            for plan, count in result.all()
        ]
