import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

from backend.db.database import get_db
from backend.modules.admin.router import router
from backend.modules.auth.dependencies import get_admin_user


class FakeAdminService:
    """Fake service for admin analytics API tests."""

    async def get_users_by_subscription_analytics(self, db):
        return {
            "total_users": 10,
            "without_subscription": 4,
            "by_plan": {
                "basic": 3,
                "pro": 2,
                "enterprise": 1,
            },
        }


class EmptyAdminService:
    """Fake service that returns empty analytics."""

    async def get_users_by_subscription_analytics(self, db):
        return {
            "total_users": 0,
            "without_subscription": 0,
            "by_plan": {
                "basic": 0,
                "pro": 0,
                "enterprise": 0,
            },
        }


def build_test_app(monkeypatch, service=None, admin_allowed: bool = True) -> FastAPI:
    app = FastAPI()
    app.include_router(router)

    async def fake_get_db():
        yield object()

    async def fake_get_admin_user():
        if not admin_allowed:
            raise HTTPException(status_code=403, detail="Admin access required")
        return object()

    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[get_admin_user] = fake_get_admin_user

    monkeypatch.setattr(
        "backend.modules.admin.router.admin_service",
        service or FakeAdminService(),
    )

    return app


@pytest.mark.asyncio
async def test_get_admin_analytics_returns_subscription_stats(monkeypatch):
    app = build_test_app(monkeypatch)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/admin/analytics")

    assert response.status_code == 200
    assert response.json() == {
        "total_users": 10,
        "without_subscription": 4,
        "by_plan": {
            "basic": 3,
            "pro": 2,
            "enterprise": 1,
        },
    }


@pytest.mark.asyncio
async def test_get_admin_analytics_response_contains_required_fields(monkeypatch):
    app = build_test_app(monkeypatch)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/admin/analytics")

    data = response.json()

    assert response.status_code == 200
    assert set(data.keys()) == {
        "total_users",
        "without_subscription",
        "by_plan",
    }


@pytest.mark.asyncio
async def test_get_admin_analytics_returns_zeroes(monkeypatch):
    app = build_test_app(monkeypatch, service=EmptyAdminService())

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/admin/analytics")

    assert response.status_code == 200
    assert response.json() == {
        "total_users": 0,
        "without_subscription": 0,
        "by_plan": {
            "basic": 0,
            "pro": 0,
            "enterprise": 0,
        },
    }


@pytest.mark.asyncio
async def test_get_admin_analytics_requires_admin(monkeypatch):
    app = build_test_app(monkeypatch, admin_allowed=False)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/admin/analytics")

    assert response.status_code == 403
    assert response.json()["detail"] == "Admin access required"


@pytest.mark.asyncio
async def test_get_admin_analytics_uses_response_model(monkeypatch):
    app = build_test_app(monkeypatch)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/admin/analytics")

    data = response.json()

    assert isinstance(data["total_users"], int)
    assert isinstance(data["without_subscription"], int)
    assert isinstance(data["by_plan"], dict)
