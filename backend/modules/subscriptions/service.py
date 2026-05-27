from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import Subscription, SubscriptionPlan, User
from backend.modules.subscriptions.interfaces import (
    BaseSubscriptionPlansProvider,
    BaseSubscriptionsRepository,
    BaseSubscriptionsService,
)
from backend.modules.subscriptions.mapper import SubscriptionMapper
from backend.modules.subscriptions.plans import SubscriptionPlansProvider
from backend.modules.subscriptions.repository import SubscriptionsRepository


class SubscriptionsService(BaseSubscriptionsService):
    """Сервис подписок пользователя."""

    def __init__(
        self,
        repository: BaseSubscriptionsRepository,
        plans_provider: BaseSubscriptionPlansProvider,
        mapper: SubscriptionMapper,
    ) -> None:
        """Инициализирует сервис подписок."""
        self.repository = repository
        self.plans_provider = plans_provider
        self.mapper = mapper

    def list_plans(self) -> list[dict]:
        """Возвращает список доступных тарифов."""
        return self.plans_provider.list_plans()

    async def get_user_subscription(self, db: AsyncSession, user_id: UUID) -> Subscription | None:
        """Возвращает активную подписку пользователя."""
        return await self.repository.get_active_by_user_id(
            db=db,
            user_id=user_id,
        )

    async def get_user_subscription_response(self, db: AsyncSession, user: User) -> dict:
        """Возвращает DTO текущей подписки пользователя."""
        subscription = await self.get_user_subscription(
            db=db,
            user_id=user.id,
        )

        if subscription is None:
            return self.mapper.empty_response(user)

        return self.mapper.active_response(subscription)

    async def set_user_subscription(self, db: AsyncSession, user: User, plan: SubscriptionPlan) -> dict:
        """Меняет подписку пользователя с удалением предыдущих подписок."""
        await self.repository.delete_by_user_id(
            db=db,
            user_id=user.id,
        )

        subscription = Subscription(
            user_id=user.id,
            plan_name=plan,
            is_active=True,
        )

        await self.repository.add(
            db=db,
            subscription=subscription,
        )

        await self.repository.commit(db)

        return await self.get_user_subscription_response(
            db=db,
            user=user,
        )

    async def user_has_paid_subscription(self, db: AsyncSession, user_id: UUID) -> bool:
        """Проверяет наличие активной подписки пользователя."""
        subscription = await self.get_user_subscription(
            db=db,
            user_id=user_id,
        )

        return subscription is not None


_subscription_plans_provider = SubscriptionPlansProvider()

subscriptions_service = SubscriptionsService(
    repository=SubscriptionsRepository(),
    plans_provider=_subscription_plans_provider,
    mapper=SubscriptionMapper(
        plans_provider=_subscription_plans_provider,
    ),
)
