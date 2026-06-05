from backend.db import SubscriptionPlan
from backend.modules.subscriptions.constants import SUBSCRIPTION_PLANS
from backend.modules.subscriptions.plans import SubscriptionPlansProvider


def test_list_plans_returns_all_configured_paid_plans():
    """Verifies that all configured paid subscription plans are returned."""

    provider = SubscriptionPlansProvider()

    result = provider.list_plans()

    assert len(result) == len(SUBSCRIPTION_PLANS)
    assert {item["plan"] for item in result} == {
        SubscriptionPlan.basic,
        SubscriptionPlan.pro,
        SubscriptionPlan.enterprise,
    }


def test_list_plans_includes_public_plan_metadata():
    """Verifies that each plan response contains API metadata fields."""

    provider = SubscriptionPlansProvider()

    result = provider.list_plans()

    for item in result:
        assert "plan" in item
        assert "title" in item
        assert "price_rub" in item
        assert "description" in item


def test_get_plan_data_returns_configured_plan_data():
    """Verifies that plan lookup returns the configured plan data."""

    provider = SubscriptionPlansProvider()

    result = provider.get_plan_data(SubscriptionPlan.pro)

    assert result == SUBSCRIPTION_PLANS[SubscriptionPlan.pro]
    assert result["title"] == "Pro"
    assert result["price_rub"] == 5000
