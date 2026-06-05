from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.logger_config import get_logger
from backend.db import SubscriptionPlan
from backend.modules.admin.interfaces import AdminAnalyticsRepositoryProtocol, BaseAdminAnalyticsService
from backend.modules.admin.repository import AdminAnalyticsRepository

logger = get_logger(__name__)

class AdminService(BaseAdminAnalyticsService):
    """Сервис административной аналитики."""

    def __init__(
        self,
        repository: AdminAnalyticsRepositoryProtocol,
    ):
        self.repository = repository

    async def get_users_by_subscription_analytics(
        self,
        db: AsyncSession,
    ) -> dict:
        """Возвращает статистику пользователей по активным подпискам."""

        logger.info(
            "Started calculating admin subscription analytics"
        )

        try:
            total_users = await self.repository.count_users(db)

            logger.info(
                "Total users received: total_users=%s",
                total_users,
            )

            by_plan = self._build_empty_plan_map()

            users_with_subscription = 0

            rows = await self.repository.count_users_by_active_subscription(
                db,
            )

            logger.info(
                "Subscription statistics received: groups=%s",
                len(rows),
            )

            for plan, count in rows:
                users_with_subscription += count
                by_plan[plan.value] = count

                logger.info(
                    "Processed subscription plan: plan=%s, users=%s",
                    plan.value,
                    count,
                )

            analytics = {
                "total_users": total_users,
                "without_subscription": max(
                    total_users - users_with_subscription,
                    0,
                ),
                "by_plan": by_plan,
            }

            logger.info(
                (
                    "Admin analytics calculated successfully: "
                    "total_users=%s, "
                    "users_with_subscription=%s, "
                    "without_subscription=%s"
                ),
                total_users,
                users_with_subscription,
                analytics["without_subscription"],
            )

            return analytics

        except Exception:
            logger.exception(
                "Failed to calculate admin subscription analytics"
            )
            raise

    def _build_empty_plan_map(self) -> dict[str, int]:
        """Создаёт словарь тарифов с нулевыми значениями."""

        plans = {
            SubscriptionPlan.basic.value: 0,
            SubscriptionPlan.pro.value: 0,
            SubscriptionPlan.enterprise.value: 0,
        }

        logger.debug(
            "Initialized empty subscription plan map"
        )

        return plans


admin_service = AdminService(
    repository=AdminAnalyticsRepository(),
)
