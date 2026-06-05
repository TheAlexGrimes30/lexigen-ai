import uuid

import pytest

from backend.db import Subscription, SubscriptionPlan, User, UserRole
from backend.modules.subscriptions.mapper import SubscriptionMapper
from backend.modules.subscriptions.plans import SubscriptionPlansProvider
from backend.modules.subscriptions.service import SubscriptionsService


class FakeSubscriptionsRepository:
    """Fake subscriptions repository for service unit tests."""

    def __init__(self, subscription: Subscription | None = None) -> None:
        """Initializes fake repository state."""

        self.subscription = subscription
        self.deleted_user_id = None
        self.added_subscription = None
        self.commit_called = False
        self.get_active_called_with = None

    async def get_active_by_user_id(self, db, user_id):
        """Returns the configured active subscription."""

        self.get_active_called_with = user_id
        return self.subscription

    async def delete_by_user_id(self, db, user_id):
        """Records that user subscriptions were deleted."""

        self.deleted_user_id = user_id
        self.subscription = None

    async def add(self, db, subscription):
        """Stores the added subscription."""

        self.added_subscription = subscription
        self.subscription = subscription
        return subscription

    async def commit(self, db):
        """Records transaction commit."""

        self.commit_called = True


def make_user(role: UserRole = UserRole.user) -> User:
    """Builds a user model for service unit tests."""

    return User(
        id=uuid.uuid4(),
        name="Test User",
        email=f"{uuid.uuid4()}@test.com",
        password_hash="hash",
        role=role,
    )


def make_subscription(
    user_id: uuid.UUID,
    plan: SubscriptionPlan = SubscriptionPlan.pro,
) -> Subscription:
    """Builds a subscription model for service unit tests."""

    return Subscription(
        id=uuid.uuid4(),
        user_id=user_id,
        plan_name=plan,
        is_active=True,
    )


def build_service(repository: FakeSubscriptionsRepository) -> SubscriptionsService:
    """Builds a subscriptions service with fake dependencies."""

    plans_provider = SubscriptionPlansProvider()

    return SubscriptionsService(
        repository=repository,
        plans_provider=plans_provider,
        mapper=SubscriptionMapper(plans_provider),
    )


def test_list_plans_delegates_to_provider():
    """Verifies that list_plans returns provider data."""

    repository = FakeSubscriptionsRepository()
    service = build_service(repository)

    result = service.list_plans()

    assert {item["plan"] for item in result} == {
        SubscriptionPlan.basic,
        SubscriptionPlan.pro,
        SubscriptionPlan.enterprise,
    }


@pytest.mark.asyncio
async def test_get_user_subscription_returns_active_subscription():
    """Verifies that an active subscription is returned by user id."""

    user = make_user()
    subscription = make_subscription(user.id)
    repository = FakeSubscriptionsRepository(subscription)
    service = build_service(repository)

    result = await service.get_user_subscription(db=None, user_id=user.id)

    assert result is subscription
    assert repository.get_active_called_with == user.id


@pytest.mark.asyncio
async def test_get_user_subscription_response_returns_empty_response():
    """Verifies that users without subscription receive an empty response."""

    user = make_user()
    repository = FakeSubscriptionsRepository(None)
    service = build_service(repository)

    result = await service.get_user_subscription_response(db=None, user=user)

    assert result["plan"] is None
    assert result["title"] == "Без подписки"
    assert result["is_active"] is False
    assert result["can_analyze_unlimited"] is False


@pytest.mark.asyncio
async def test_get_user_subscription_response_returns_active_response():
    """Verifies that active subscription is converted to response data."""

    user = make_user()
    subscription = make_subscription(user.id, SubscriptionPlan.pro)
    repository = FakeSubscriptionsRepository(subscription)
    service = build_service(repository)

    result = await service.get_user_subscription_response(db=None, user=user)

    assert result["plan"] == SubscriptionPlan.pro
    assert result["title"] == "Pro"
    assert result["price_rub"] == 5000
    assert result["is_active"] is True


@pytest.mark.asyncio
async def test_set_user_subscription_replaces_existing_subscription():
    """Verifies that setting a subscription deletes previous records and adds a new one."""

    user = make_user()
    repository = FakeSubscriptionsRepository(make_subscription(user.id, SubscriptionPlan.basic))
    service = build_service(repository)

    result = await service.set_user_subscription(
        db=None,
        user=user,
        plan=SubscriptionPlan.enterprise,
    )

    assert repository.deleted_user_id == user.id
    assert repository.added_subscription is not None
    assert repository.added_subscription.user_id == user.id
    assert repository.added_subscription.plan_name == SubscriptionPlan.enterprise
    assert repository.added_subscription.is_active is True
    assert repository.commit_called is True
    assert result["plan"] == SubscriptionPlan.enterprise


@pytest.mark.asyncio
async def test_user_has_paid_subscription_returns_true_when_subscription_exists():
    """Verifies that paid subscription check is true when subscription exists."""

    user = make_user()
    repository = FakeSubscriptionsRepository(make_subscription(user.id))
    service = build_service(repository)

    result = await service.user_has_paid_subscription(db=None, user_id=user.id)

    assert result is True


@pytest.mark.asyncio
async def test_user_has_paid_subscription_returns_false_without_subscription():
    """Verifies that paid subscription check is false without active subscription."""

    user = make_user()
    repository = FakeSubscriptionsRepository(None)
    service = build_service(repository)

    result = await service.user_has_paid_subscription(db=None, user_id=user.id)

    assert result is False
