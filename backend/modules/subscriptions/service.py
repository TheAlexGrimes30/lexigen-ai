from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.logger_config import get_logger
from backend.db import Subscription, SubscriptionPlan, User
from backend.modules.subscriptions.interfaces import (
    BaseSubscriptionPlansProvider,
    BaseSubscriptionsRepository,
    BaseSubscriptionsService,
)
from backend.modules.subscriptions.mapper import SubscriptionMapper
from backend.modules.subscriptions.plans import SubscriptionPlansProvider
from backend.modules.subscriptions.repository import SubscriptionsRepository


logger = get_logger(__name__)


class SubscriptionsService(BaseSubscriptionsService):
    """Сервис подписок пользователя."""

    def __init__(
        self,
        repository: BaseSubscriptionsRepository,
        plans_provider: BaseSubscriptionPlansProvider,
        mapper: SubscriptionMapper,
    ):
        self.repository = repository
        self.plans_provider = plans_provider
        self.mapper = mapper

    def list_plans(self) -> list[dict]:
        """Возвращает список доступных тарифов."""

        logger.info(
            "Subscription plans requested"
        )

        plans = self.plans_provider.list_plans()

        logger.info(
            "Subscription plans returned: count=%s",
            len(plans),
        )

        return plans

    async def get_user_subscription(
        self,
        db: AsyncSession,
        user_id: UUID,
    ) -> Subscription | None:
        """Возвращает активную подписку пользователя."""

        logger.info(
            "Getting user subscription: user_id=%s",
            user_id,
        )

        return await self.repository.get_active_by_user_id(
            db=db,
            user_id=user_id,
        )

    async def get_user_subscription_response(
        self,
        db: AsyncSession,
        user: User,
    ) -> dict:
        """Возвращает DTO текущей подписки пользователя."""

        logger.info(
            "Building subscription response: user_id=%s",
            user.id,
        )

        subscription = await self.get_user_subscription(
            db=db,
            user_id=user.id,
        )

        if subscription is None:

            logger.info(
                "User has no active subscription: user_id=%s",
                user.id,
            )

            return self.mapper.empty_response(user)

        logger.info(
            "User has active subscription: user_id=%s, plan=%s",
            user.id,
            subscription.plan_name.value,
        )

        return self.mapper.active_response(
            subscription,
        )

    async def set_user_subscription(
        self,
        db: AsyncSession,
        user: User,
        plan: SubscriptionPlan,
    ) -> dict:
        """Меняет подписку пользователя."""

        logger.info(
            "Subscription change started: user_id=%s, plan=%s",
            user.id,
            plan.value,
        )

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

        logger.info(
            "Subscription changed successfully: user_id=%s, plan=%s",
            user.id,
            plan.value,
        )

        return await self.get_user_subscription_response(
            db=db,
            user=user,
        )

    async def user_has_paid_subscription(
        self,
        db: AsyncSession,
        user_id: UUID,
    ) -> bool:
        """Проверяет наличие активной подписки."""

        subscription = await self.get_user_subscription(
            db=db,
            user_id=user_id,
        )

        result = subscription is not None

        logger.info(
            "Subscription access check: user_id=%s, has_subscription=%s",
            user_id,
            result,
        )

        return result


_subscription_plans_provider = SubscriptionPlansProvider()

subscriptions_service = SubscriptionsService(
    repository=SubscriptionsRepository(),
    plans_provider=_subscription_plans_provider,
    mapper=SubscriptionMapper(
        plans_provider=_subscription_plans_provider,
    ),
)
