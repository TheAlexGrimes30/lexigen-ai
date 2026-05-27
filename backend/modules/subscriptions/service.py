from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import SubscriptionPlan, Subscription, User, UserRole

SUBSCRIPTION_PLANS = {
    SubscriptionPlan.basic: {
        "title": "Basic",
        "price_rub": 1000,
        "description": "Базовый доступ к анализу документов.",
    },
    SubscriptionPlan.pro: {
        "title": "Pro",
        "price_rub": 5000,
        "description": "Расширенный доступ для регулярной работы с договорами.",
    },
    SubscriptionPlan.enterprise: {
        "title": "Enterprise",
        "price_rub": 20000,
        "description": "Корпоративный тариф для команд и бизнеса.",
    },
}


class SubscriptionsService:
    """Сервис подписок и аналитики по подпискам."""

    def list_plans(self) -> list[dict]:
        return [
            {
                "plan": plan,
                **data,
            }
            for plan, data in SUBSCRIPTION_PLANS.items()
        ]

    async def get_user_subscription(
        self,
        db: AsyncSession,
        user_id,
    ) -> Subscription | None:
        stmt = select(Subscription).where(
            Subscription.user_id == user_id,
            Subscription.is_active.is_(True),
        )

        result = await db.execute(stmt)

        return result.scalar_one_or_none()

    async def get_user_subscription_response(
        self,
        db: AsyncSession,
        user: User,
    ) -> dict:
        subscription = await self.get_user_subscription(
            db,
            user.id,
        )

        if subscription is None:
            return {
                "plan": None,
                "title": "Без подписки",
                "price_rub": 0,
                "is_active": False,
                "can_analyze_unlimited": user.role == UserRole.admin,
            }

        plan_data = SUBSCRIPTION_PLANS[subscription.plan_name]

        return {
            "plan": subscription.plan_name,
            "title": plan_data["title"],
            "price_rub": plan_data["price_rub"],
            "is_active": subscription.is_active,
            "can_analyze_unlimited": True,
        }

    async def set_user_subscription(
        self,
        db: AsyncSession,
        user: User,
        plan: SubscriptionPlan,
    ) -> dict:
        subscription = await self.get_user_subscription(
            db,
            user.id,
        )

        if subscription is None:
            subscription = Subscription(
                user_id=user.id,
                plan_name=plan,
                is_active=True,
            )

            db.add(subscription)
        else:
            subscription.plan_name = plan
            subscription.is_active = True

        await db.commit()

        return await self.get_user_subscription_response(
            db,
            user,
        )

    async def user_has_paid_subscription(
        self,
        db: AsyncSession,
        user_id,
    ) -> bool:
        subscription = await self.get_user_subscription(
            db,
            user_id,
        )

        return subscription is not None

    async def get_admin_analytics(
        self,
        db: AsyncSession,
    ) -> dict:
        total_users = await db.scalar(
            select(func.count(User.id))
        )

        by_plan = {
            SubscriptionPlan.basic.value: 0,
            SubscriptionPlan.pro.value: 0,
            SubscriptionPlan.enterprise.value: 0,
        }

        stmt = (
            select(
                Subscription.plan_name,
                func.count(Subscription.id),
            )
            .where(Subscription.is_active.is_(True))
            .group_by(Subscription.plan_name)
        )

        result = await db.execute(stmt)

        users_with_subscription = 0

        for plan, count in result.all():
            count = int(count)
            users_with_subscription += count
            by_plan[plan.value] = count

        total_users = int(total_users or 0)

        return {
            "total_users": total_users,
            "without_subscription": max(
                total_users - users_with_subscription,
                0,
            ),
            "by_plan": by_plan,
        }


subscriptions_service = SubscriptionsService()
