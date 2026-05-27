from abc import ABC, abstractmethod
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession


class AdminAnalyticsRepositoryProtocol(Protocol):
    """Протокол репозитория административной аналитики."""

    async def count_users(
        self,
        db: AsyncSession,
    ) -> int:
        """Возвращает общее количество пользователей."""

        ...

    async def count_users_by_active_subscription(
        self,
        db: AsyncSession,
    ) -> list[tuple[object, int]]:
        """Возвращает количество пользователей по активным подпискам."""

        ...


class BaseAdminAnalyticsService(ABC):
    """Абстрактный сервис административной аналитики."""

    @abstractmethod
    async def get_users_by_subscription_analytics(
        self,
        db: AsyncSession,
    ) -> dict:
        """Возвращает аналитику пользователей по подпискам."""

        raise NotImplementedError