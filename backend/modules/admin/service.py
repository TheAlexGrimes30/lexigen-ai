from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import SubscriptionPlan
from backend.modules.admin.interfaces import AdminAnalyticsRepositoryProtocol, BaseAdminAnalyticsService
from backend.modules.admin.repository import AdminAnalyticsRepository


class AdminService(BaseAdminAnalyticsService):
    """Сервис административной аналитики."""

    def __init__(
        self,
        repository: AdminAnalyticsRepositoryProtocol,
    ) -> None:
        """Инициализирует сервис административной аналитики."""

        self.repository = repository

    async def get_users_by_subscription_analytics(
        self,
        db: AsyncSession,
    ) -> dict:
        """Возвращает статистику пользователей по активным подпискам."""

        total_users = await self.repository.count_users(db)

        by_plan = self._build_empty_plan_map()

        users_with_subscription = 0

        rows = await self.repository.count_users_by_active_subscription(db)

        for plan, count in rows:
            users_with_subscription += count
            by_plan[plan.value] = count

        return {
            "total_users": total_users,
            "without_subscription": max(
                total_users - users_with_subscription,
                0,
            ),
            "by_plan": by_plan,
        }

    def _build_empty_plan_map(self) -> dict[str, int]:
        """Создаёт словарь тарифов с нулевыми значениями."""

        return {
            SubscriptionPlan.basic.value: 0,
            SubscriptionPlan.pro.value: 0,
            SubscriptionPlan.enterprise.value: 0,
        }


admin_service = AdminService(
    repository=AdminAnalyticsRepository(),
)
