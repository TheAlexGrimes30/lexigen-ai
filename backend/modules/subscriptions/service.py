from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import Subscription, SubscriptionPlan, User, UserRole
from backend.modules.subscriptions.constants import SUBSCRIPTION_PLANS


class SubscriptionsService:
    """Сервис подписок пользователя."""

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
        """
        Меняет подписку пользователя.

        Старые подписки физически удаляются, затем создаётся новая активная.
        """

        await db.execute(
            delete(Subscription).where(
                Subscription.user_id == user.id,
            )
        )

        subscription = Subscription(
            user_id=user.id,
            plan_name=plan,
            is_active=True,
        )

        db.add(subscription)

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


subscriptions_service = SubscriptionsService()
