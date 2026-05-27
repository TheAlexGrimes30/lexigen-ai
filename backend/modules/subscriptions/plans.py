from backend.db import SubscriptionPlan
from backend.modules.subscriptions.constants import SUBSCRIPTION_PLANS
from backend.modules.subscriptions.interfaces import BaseSubscriptionPlansProvider


class SubscriptionPlansProvider(BaseSubscriptionPlansProvider):
    """Источник доступных тарифных планов."""

    def list_plans(self) -> list[dict]:
        """Возвращает список тарифов в формате API."""
        return [
            {
                "plan": plan,
                **data,
            }
            for plan, data in SUBSCRIPTION_PLANS.items()
        ]

    def get_plan_data(self, plan: SubscriptionPlan) -> dict:
        """Возвращает данные тарифа по его enum-значению."""
        return SUBSCRIPTION_PLANS[plan]
