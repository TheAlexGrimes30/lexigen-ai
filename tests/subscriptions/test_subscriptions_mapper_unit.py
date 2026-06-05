import uuid

from backend.db import Subscription, SubscriptionPlan, User, UserRole
from backend.modules.subscriptions.mapper import SubscriptionMapper
from backend.modules.subscriptions.plans import SubscriptionPlansProvider


def make_user(role: UserRole = UserRole.user) -> User:
    """Builds a user model for mapper unit tests."""

    return User(
        id=uuid.uuid4(),
        name="Test User",
        email=f"{uuid.uuid4()}@test.com",
        password_hash="hash",
        role=role,
    )


def make_subscription(
    user_id: uuid.UUID,
    plan: SubscriptionPlan = SubscriptionPlan.basic,
    is_active: bool = True,
) -> Subscription:
    """Builds a subscription model for mapper unit tests."""

    return Subscription(
        id=uuid.uuid4(),
        user_id=user_id,
        plan_name=plan,
        is_active=is_active,
    )


def test_empty_response_for_regular_user_disables_unlimited_analysis():
    """Verifies that a regular user without subscription has no unlimited analysis."""

    mapper = SubscriptionMapper(SubscriptionPlansProvider())
    user = make_user(UserRole.user)

    result = mapper.empty_response(user)

    assert result == {
        "plan": None,
        "title": "Без подписки",
        "price_rub": 0,
        "is_active": False,
        "can_analyze_unlimited": False,
    }


def test_empty_response_for_admin_enables_unlimited_analysis():
    """Verifies that an admin without subscription still has unlimited analysis."""

    mapper = SubscriptionMapper(SubscriptionPlansProvider())
    user = make_user(UserRole.admin)

    result = mapper.empty_response(user)

    assert result["plan"] is None
    assert result["is_active"] is False
    assert result["can_analyze_unlimited"] is True


def test_active_response_maps_subscription_plan_metadata():
    """Verifies that active subscription data is mapped to response metadata."""

    mapper = SubscriptionMapper(SubscriptionPlansProvider())
    user = make_user()
    subscription = make_subscription(user.id, SubscriptionPlan.enterprise)

    result = mapper.active_response(subscription)

    assert result["plan"] == SubscriptionPlan.enterprise
    assert result["title"] == "Enterprise"
    assert result["price_rub"] == 20000
    assert result["is_active"] is True
    assert result["can_analyze_unlimited"] is True


def test_active_response_preserves_inactive_flag():
    """Verifies that active_response preserves the subscription active flag."""

    mapper = SubscriptionMapper(SubscriptionPlansProvider())
    user = make_user()
    subscription = make_subscription(
        user_id=user.id,
        plan=SubscriptionPlan.basic,
        is_active=False,
    )

    result = mapper.active_response(subscription)

    assert result["plan"] == SubscriptionPlan.basic
    assert result["is_active"] is False
