import pytest

from backend.db import SubscriptionPlan
from backend.modules.admin.service import AdminService


class FakeAdminAnalyticsRepository:
    """Fake repository for AdminService unit tests."""

    def __init__(self, total_users: int, rows: list[tuple[object, int]]) -> None:
        self.total_users = total_users
        self.rows = rows
        self.count_users_called = False
        self.count_users_by_active_subscription_called = False

    async def count_users(self, db):
        self.count_users_called = True
        return self.total_users

    async def count_users_by_active_subscription(self, db):
        self.count_users_by_active_subscription_called = True
        return self.rows


@pytest.mark.asyncio
async def test_get_users_by_subscription_analytics_counts_without_subscription():
    repository = FakeAdminAnalyticsRepository(
        total_users=10,
        rows=[
            (SubscriptionPlan.basic, 3),
            (SubscriptionPlan.pro, 2),
        ],
    )
    service = AdminService(repository=repository)

    result = await service.get_users_by_subscription_analytics(db=None)

    assert result == {
        "total_users": 10,
        "without_subscription": 5,
        "by_plan": {
            SubscriptionPlan.basic.value: 3,
            SubscriptionPlan.pro.value: 2,
            SubscriptionPlan.enterprise.value: 0,
        },
    }
    assert repository.count_users_called is True
    assert repository.count_users_by_active_subscription_called is True


@pytest.mark.asyncio
async def test_get_users_by_subscription_analytics_never_returns_negative_without_subscription():
    repository = FakeAdminAnalyticsRepository(
        total_users=2,
        rows=[
            (SubscriptionPlan.basic, 3),
            (SubscriptionPlan.pro, 2),
        ],
    )
    service = AdminService(repository=repository)

    result = await service.get_users_by_subscription_analytics(db=None)

    assert result["total_users"] == 2
    assert result["without_subscription"] == 0


@pytest.mark.asyncio
async def test_get_users_by_subscription_analytics_returns_zeroes_when_no_subscriptions():
    repository = FakeAdminAnalyticsRepository(
        total_users=4,
        rows=[],
    )
    service = AdminService(repository=repository)

    result = await service.get_users_by_subscription_analytics(db=None)

    assert result == {
        "total_users": 4,
        "without_subscription": 4,
        "by_plan": {
            SubscriptionPlan.basic.value: 0,
            SubscriptionPlan.pro.value: 0,
            SubscriptionPlan.enterprise.value: 0,
        },
    }
