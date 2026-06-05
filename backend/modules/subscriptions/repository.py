from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.logger_config import get_logger
from backend.db import Subscription
from backend.modules.subscriptions.interfaces import BaseSubscriptionsRepository

logger = get_logger(__name__)


class SubscriptionsRepository(BaseSubscriptionsRepository):
    """Репозиторий операций с подписками."""

    async def get_active_by_user_id(
        self,
        db: AsyncSession,
        user_id: UUID,
    ) -> Subscription | None:
        """Возвращает активную подписку пользователя."""

        logger.info(
            "Fetching active subscription: user_id=%s",
            user_id,
        )

        try:
            stmt = select(Subscription).where(
                Subscription.user_id == user_id,
                Subscription.is_active.is_(True),
            )

            result = await db.execute(stmt)

            subscription = result.scalar_one_or_none()

            logger.info(
                "Active subscription fetched: user_id=%s, found=%s",
                user_id,
                subscription is not None,
            )

            return subscription

        except Exception:
            logger.exception(
                "Failed to fetch active subscription: user_id=%s",
                user_id,
            )
            raise

    async def delete_by_user_id(
        self,
        db: AsyncSession,
        user_id: UUID,
    ) -> None:
        """Физически удаляет все подписки пользователя."""

        logger.info(
            "Deleting subscriptions: user_id=%s",
            user_id,
        )

        try:
            await db.execute(
                delete(Subscription).where(
                    Subscription.user_id == user_id,
                )
            )

            logger.info(
                "Subscriptions deleted successfully: user_id=%s",
                user_id,
            )

        except Exception:
            logger.exception(
                "Failed to delete subscriptions: user_id=%s",
                user_id,
            )
            raise

    async def add(
        self,
        db: AsyncSession,
        subscription: Subscription,
    ) -> Subscription:
        """Добавляет подписку в текущую сессию."""

        logger.info(
            "Adding subscription: user_id=%s, plan=%s",
            subscription.user_id,
            subscription.plan_name.value,
        )

        try:
            db.add(subscription)

            logger.info(
                "Subscription added to session: user_id=%s",
                subscription.user_id,
            )

            return subscription

        except Exception:
            logger.exception(
                "Failed to add subscription: user_id=%s",
                subscription.user_id,
            )
            raise

    async def commit(
        self,
        db: AsyncSession,
    ) -> None:
        """Фиксирует текущую транзакцию."""

        logger.info("Committing subscription transaction")

        try:
            await db.commit()

            logger.info(
                "Subscription transaction committed successfully"
            )

        except Exception:
            logger.exception(
                "Failed to commit subscription transaction"
            )
            raise