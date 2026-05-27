from backend.db import Subscription, User, UserRole
from backend.modules.subscriptions.interfaces import BaseSubscriptionPlansProvider


class SubscriptionMapper:
    """Маппер подписок в DTO-словарь."""

    def __init__(self, plans_provider: BaseSubscriptionPlansProvider) -> None:
        """Инициализирует маппер подписок."""
        self.plans_provider = plans_provider

    def empty_response(self, user: User) -> dict:
        """Возвращает DTO для пользователя без подписки."""
        return {
            "plan": None,
            "title": "Без подписки",
            "price_rub": 0,
            "is_active": False,
            "can_analyze_unlimited": user.role == UserRole.admin,
        }

    def active_response(self, subscription: Subscription) -> dict:
        """Возвращает DTO активной подписки пользователя."""
        plan_data = self.plans_provider.get_plan_data(
            subscription.plan_name
        )

        return {
            "plan": subscription.plan_name,
            "title": plan_data["title"],
            "price_rub": plan_data["price_rub"],
            "is_active": subscription.is_active,
            "can_analyze_unlimited": True,
        }
