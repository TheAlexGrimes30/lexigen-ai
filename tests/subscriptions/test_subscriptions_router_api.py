import uuid

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

from backend.db import SubscriptionPlan, User, UserRole
from backend.db.database import get_db
from backend.modules.auth.dependencies import get_current_user
from backend.modules.subscriptions.router import router


class FakeSubscriptionsService:
    """Fake subscriptions service for API tests."""

    def __init__(self) -> None:
        """Initializes fake service state."""

        self.updated_plan = None

    def list_plans(self) -> list[dict]:
        """Returns fixed subscription plans."""

        return [
            {
                "plan": SubscriptionPlan.basic,
                "title": "Basic",
                "price_rub": 1000,
                "description": "Basic plan",
            },
            {
                "plan": SubscriptionPlan.pro,
                "title": "Pro",
                "price_rub": 5000,
                "description": "Pro plan",
            },
        ]

    async def get_user_subscription_response(self, db, user):
        """Returns current user subscription response."""

        return {
            "plan": SubscriptionPlan.basic,
            "title": "Basic",
            "price_rub": 1000,
            "is_active": True,
            "can_analyze_unlimited": True,
        }

    async def set_user_subscription(self, db, user, plan):
        """Stores selected plan and returns updated subscription response."""

        self.updated_plan = plan

        return {
            "plan": plan,
            "title": "Pro",
            "price_rub": 5000,
            "is_active": True,
            "can_analyze_unlimited": True,
        }


def make_user() -> User:
    """Builds a user model for API tests."""

    return User(
        id=uuid.uuid4(),
        name="Test User",
        email="user@test.com",
        password_hash="hash",
        role=UserRole.user,
    )


def build_test_app(monkeypatch, service=None, authenticated: bool = True) -> FastAPI:
    """Builds a FastAPI app with overridden dependencies."""

    app = FastAPI()
    app.include_router(router)

    async def fake_get_db():
        """Returns a fake database dependency."""
        yield object()

    async def fake_get_current_user():
        """Returns a fake current user or raises HTTP 401."""
        if not authenticated:
            raise HTTPException(status_code=401, detail="Требуется авторизация")
        return make_user()

    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[get_current_user] = fake_get_current_user

    monkeypatch.setattr(
        "backend.modules.subscriptions.router.subscriptions_service",
        service or FakeSubscriptionsService(),
    )

    return app


@pytest.mark.asyncio
async def test_get_subscription_plans_returns_available_plans(monkeypatch):
    """Verifies that GET /plans returns available subscription plans."""

    app = build_test_app(monkeypatch)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/subscriptions/plans")

    assert response.status_code == 200
    assert response.json() == [
        {
            "plan": "basic",
            "title": "Basic",
            "price_rub": 1000,
            "description": "Basic plan",
        },
        {
            "plan": "pro",
            "title": "Pro",
            "price_rub": 5000,
            "description": "Pro plan",
        },
    ]


@pytest.mark.asyncio
async def test_get_my_subscription_returns_current_subscription(monkeypatch):
    """Verifies that GET /me returns current user subscription."""

    app = build_test_app(monkeypatch)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/subscriptions/me")

    assert response.status_code == 200
    assert response.json() == {
        "plan": "basic",
        "title": "Basic",
        "price_rub": 1000,
        "is_active": True,
        "can_analyze_unlimited": True,
    }


@pytest.mark.asyncio
async def test_get_my_subscription_requires_authentication(monkeypatch):
    """Verifies that GET /me requires an authenticated user."""

    app = build_test_app(monkeypatch, authenticated=False)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/subscriptions/me")

    assert response.status_code == 401
    assert response.json()["detail"] == "Требуется авторизация"


@pytest.mark.asyncio
async def test_put_my_subscription_updates_subscription(monkeypatch):
    """Verifies that PUT /me updates current user subscription."""

    service = FakeSubscriptionsService()
    app = build_test_app(monkeypatch, service=service)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.put(
            "/api/subscriptions/me",
            json={"plan": "pro"},
        )

    assert response.status_code == 200
    assert service.updated_plan == SubscriptionPlan.pro
    assert response.json()["plan"] == "pro"


@pytest.mark.asyncio
async def test_put_my_subscription_rejects_invalid_plan(monkeypatch):
    """Verifies that PUT /me rejects invalid subscription plan values."""

    app = build_test_app(monkeypatch)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.put(
            "/api/subscriptions/me",
            json={"plan": "invalid"},
        )

    assert response.status_code == 422
