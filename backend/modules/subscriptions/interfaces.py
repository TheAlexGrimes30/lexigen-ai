from abc import ABC, abstractmethod
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import Subscription, SubscriptionPlan, User


class BaseSubscriptionPlansProvider(ABC):
    """Абстракция источника тарифных планов."""

    @abstractmethod
    def list_plans(self) -> list[dict]:
        """Возвращает список доступных тарифных планов."""
        raise NotImplementedError

    @abstractmethod
    def get_plan_data(self, plan: SubscriptionPlan) -> dict:
        """Возвращает данные тарифного плана."""
        raise NotImplementedError


class BaseSubscriptionsRepository(ABC):
    """Абстрактный репозиторий подписок."""

    @abstractmethod
    async def get_active_by_user_id(self, db: AsyncSession, user_id: UUID) -> Subscription | None:
        """Возвращает активную подписку пользователя."""
        raise NotImplementedError

    @abstractmethod
    async def delete_by_user_id(self, db: AsyncSession, user_id: UUID) -> None:
        """Удаляет подписки пользователя."""
        raise NotImplementedError

    @abstractmethod
    async def add(self, db: AsyncSession, subscription: Subscription) -> Subscription:
        """Добавляет подписку в сессию."""
        raise NotImplementedError

    @abstractmethod
    async def commit(self, db: AsyncSession) -> None:
        """Фиксирует транзакцию."""
        raise NotImplementedError


class BaseSubscriptionsService(ABC):
    """Абстрактный сервис подписок."""

    @abstractmethod
    def list_plans(self) -> list[dict]:
        """Возвращает список доступных тарифных планов."""
        raise NotImplementedError

    @abstractmethod
    async def get_user_subscription(self, db: AsyncSession, user_id: UUID) -> Subscription | None:
        """Возвращает активную подписку пользователя."""
        raise NotImplementedError

    @abstractmethod
    async def get_user_subscription_response(self, db: AsyncSession, user: User) -> dict:
        """Возвращает DTO текущей подписки пользователя."""
        raise NotImplementedError

    @abstractmethod
    async def set_user_subscription(self, db: AsyncSession, user: User, plan: SubscriptionPlan) -> dict:
        """Меняет подписку пользователя."""
        raise NotImplementedError

    @abstractmethod
    async def user_has_paid_subscription(self, db: AsyncSession, user_id: UUID) -> bool:
        """Проверяет наличие активной платной подписки."""
        raise NotImplementedError
