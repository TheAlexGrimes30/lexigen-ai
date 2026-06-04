import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

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


def build_test_app(monkeypatch) -> FastAPI:
    app = FastAPI()
    app.include_router(router)

    async def fake_get_db():
        yield object()

    async def fake_get_admin_user():
        return object()

    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[get_admin_user] = fake_get_admin_user

    monkeypatch.setattr(
        "backend.modules.admin.router.admin_service",
        FakeAdminService(),
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
async def test_get_admin_analytics_requires_admin_when_dependency_not_overridden(monkeypatch):
    app = FastAPI()
    app.include_router(router)

    async def fake_get_db():
        yield object()

    app.dependency_overrides[get_db] = fake_get_db

    monkeypatch.setattr(
        "backend.modules.admin.router.admin_service",
        FakeAdminService(),
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/admin/analytics")

    assert response.status_code in {401, 403}
