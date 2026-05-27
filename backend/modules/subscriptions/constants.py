from backend.db import SubscriptionPlan

SUBSCRIPTION_PLANS = {
    SubscriptionPlan.basic: {
        "title": "Basic",
        "price_rub": 1000,
        "description": "Базовый доступ к анализу документов.",
    },
    SubscriptionPlan.pro: {
        "title": "Pro",
        "price_rub": 5000,
        "description": "Расширенный доступ для регулярной работы с договорами.",
    },
    SubscriptionPlan.enterprise: {
        "title": "Enterprise",
        "price_rub": 20000,
        "description": "Корпоративный тариф для команд и бизнеса.",
    },
}
