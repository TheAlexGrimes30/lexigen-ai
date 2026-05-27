from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import User, SubscriptionPlan, Subscription


class AdminService:
    """Сервис административной аналитики."""

    async def get_users_by_subscription_analytics(
        self,
        db: AsyncSession,
    ) -> dict:
        total_users = int(
            await db.scalar(select(func.count(User.id)))
            or 0
        )

        by_plan = {
            SubscriptionPlan.basic.value: 0,
            SubscriptionPlan.pro.value: 0,
            SubscriptionPlan.enterprise.value: 0,
        }

        stmt = (
            select(
                Subscription.plan_name,
                func.count(func.distinct(Subscription.user_id)),
            )
            .where(Subscription.is_active.is_(True))
            .group_by(Subscription.plan_name)
        )

        result = await db.execute(stmt)

        users_with_subscription = 0

        for plan, count in result.all():
            count = int(count or 0)
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


admin_service = AdminService()