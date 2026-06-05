from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.logger_config import get_logger
from backend.db import User, Subscription

logger = get_logger(__name__)

class AdminAnalyticsRepository:
    """Репозиторий административной аналитики."""

    async def count_users(
        self,
        db: AsyncSession,
    ) -> int:
        """Возвращает общее количество зарегистрированных пользователей."""

        try:
            total_users = await db.scalar(
                select(func.count(User.id))
            )

            result = int(total_users or 0)

            logger.info(
                "Counted registered users: total_users=%s",
                result,
            )

            return result

        except Exception:
            logger.exception(
                "Failed to count registered users"
            )
            raise

    async def count_users_by_active_subscription(
        self,
        db: AsyncSession,
    ) -> list[tuple[object, int]]:
        """Возвращает количество пользователей по активным подпискам."""

        try:
            stmt = (
                select(
                    Subscription.plan_name,
                    func.count(func.distinct(Subscription.user_id)),
                )
                .where(Subscription.is_active.is_(True))
                .group_by(Subscription.plan_name)
            )

            result = await db.execute(stmt)

            rows = [
                (
                    plan,
                    int(count or 0),
                )
                for plan, count in result.all()
            ]

            logger.info(
                "Counted users by active subscription: groups=%s",
                len(rows),
            )

            return rows

        except Exception:
            logger.exception(
                "Failed to count users by active subscription"
            )
            raise