import uuid
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException, FastAPI
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import UserRole, User
from backend.db.database import get_db
from backend.modules.auth.dependencies import get_admin_user, get_current_user
from backend.modules.auth.router import router
from backend.modules.auth.schema import AdminAnalyticsResponse, AuthUserResponse


class FakeAuthService:
    """Fake auth service for auth router API tests."""

    def __init__(self) -> None:
        self.registered_user = make_user(name="Registered User", email="registered@test.com")
        self.logged_in_user = make_user(name="Login User", email="login@test.com")
        self.promoted_user = make_user(name="Promoted User", email="promoted@test.com", role=UserRole.admin)
        self.analytics = AdminAnalyticsResponse(users_count=2, chats_count=3, messages_count=5)
        self.register_payload: tuple[str, str, str] | None = None
        self.login_payload: tuple[str, str] | None = None
        self.promoted_input_user: User | None = None

    async def register(self, db: AsyncSession, name: str, email: str, password: str) -> User:
        """Returns a deterministic registered user."""

        self.register_payload = (name, email, password)
        self.registered_user.name = name
        self.registered_user.email = email.lower()
        return self.registered_user

    async def login(self, db: AsyncSession, email: str, password: str) -> User:
        """Returns a deterministic logged-in user."""

        self.login_payload = (email, password)
        return self.logged_in_user

    async def promote_to_admin(self, db: AsyncSession, current_user: User) -> User:
        """Returns a deterministic promoted admin user."""

        self.promoted_input_user = current_user
        self.promoted_user.id = current_user.id
        return self.promoted_user

    async def get_admin_analytics(self, db: AsyncSession) -> AdminAnalyticsResponse:
        """Returns deterministic admin analytics."""

        return self.analytics

    def create_access_token(self, user: User) -> str:
        """Returns a deterministic access token for the user."""

        return f"token:{user.id}"

    def to_auth_user(self, user: User) -> AuthUserResponse:
        """Converts a user model into an API response DTO."""

        return AuthUserResponse(
            id=user.id,
            name=user.name,
            email=user.email,
            role=user.role.value,
            created_at=user.created_at,
        )

def make_user(
    *,
    name: str = "Test User",
    email: str = "user@test.com",
    role: UserRole = UserRole.user,
) -> User:
    """Builds a user model for auth router API tests."""

    user = User(
        id=uuid.uuid4(),
        name=name,
        email=email,
        password_hash="hash",
        role=role,
    )
    user.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return user

def build_test_app(
    monkeypatch,
    *,
    service: FakeAuthService | None = None,
    current_user: User | None = None,
    admin_allowed: bool = True,
) -> FastAPI:
    """Creates a FastAPI test app with overridden dependencies and auth service."""

    app = FastAPI()
    app.include_router(router)

    fake_service = service or FakeAuthService()
    fake_current_user = current_user or make_user()

    async def fake_get_db():
        """Returns a placeholder database dependency for API tests."""

        yield object()

    async def fake_get_current_user():
        """Returns the configured authenticated user."""

        return fake_current_user

    async def fake_get_admin_user():
        """Returns an admin user or raises HTTP 403 for forbidden scenarios."""

        if not admin_allowed:
            raise HTTPException(status_code=403, detail="Доступ только для администраторов")
        return make_user(role=UserRole.admin)

    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[get_current_user] = fake_get_current_user
    app.dependency_overrides[get_admin_user] = fake_get_admin_user

    monkeypatch.setattr("backend.modules.auth.router.auth_service", fake_service)

    return app

@pytest.mark.asyncio
async def test_register_returns_token_and_user(monkeypatch):
    """Verifies that registration returns an access token and created user data."""

    service = FakeAuthService()
    app = build_test_app(monkeypatch, service=service)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/auth/register",
            json={
                "name": "John Doe",
                "email": "JOHN@TEST.COM",
                "password": "secret123",
                "password_confirm": "secret123",
            },
        )

    data = response.json()

    assert response.status_code == 200
    assert data["access_token"].startswith("token:")
    assert data["token_type"] == "bearer"
    assert data["user"]["name"] == "John Doe"
    assert data["user"]["email"] == "john@test.com"
    assert service.register_payload == ("John Doe", "JOHN@TEST.COM", "secret123")

@pytest.mark.asyncio
async def test_register_rejects_password_mismatch(monkeypatch):
    """Verifies that registration returns HTTP 400 when passwords do not match."""

    app = build_test_app(monkeypatch)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/auth/register",
            json={
                "name": "John Doe",
                "email": "john@test.com",
                "password": "secret123",
                "password_confirm": "another123",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Пароли не совпадают"

@pytest.mark.asyncio
async def test_register_validates_short_password(monkeypatch):
    """Verifies that registration request validation rejects short passwords."""

    app = build_test_app(monkeypatch)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/auth/register",
            json={
                "name": "John Doe",
                "email": "john@test.com",
                "password": "123",
                "password_confirm": "123",
            },
        )

    assert response.status_code == 422

@pytest.mark.asyncio
async def test_login_returns_token_and_user(monkeypatch):
    """Verifies that login returns an access token and authenticated user data."""

    service = FakeAuthService()
    app = build_test_app(monkeypatch, service=service)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/auth/login",
            json={"email": "login@test.com", "password": "secret123"},
        )

    data = response.json()

    assert response.status_code == 200
    assert data["access_token"].startswith("token:")
    assert data["user"]["email"] == "login@test.com"
    assert service.login_payload == ("login@test.com", "secret123")

@pytest.mark.asyncio
async def test_login_validates_short_password(monkeypatch):
    """Verifies that login request validation rejects short passwords."""

    app = build_test_app(monkeypatch)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/auth/login",
            json={"email": "login@test.com", "password": "123"},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_me_returns_current_user(monkeypatch):
    """Verifies that the /me endpoint returns the current authenticated user."""

    current_user = make_user(name="Current User", email="current@test.com")
    app = build_test_app(monkeypatch, current_user=current_user)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/auth/me")

    data = response.json()

    assert response.status_code == 200
    assert data["id"] == str(current_user.id)
    assert data["email"] == "current@test.com"
    assert data["role"] == "user"


@pytest.mark.asyncio
async def test_become_admin_returns_new_token_and_admin_user(monkeypatch):
    """Verifies that become-admin promotes the current user and returns a new token."""

    current_user = make_user(email="regular@test.com", role=UserRole.user)
    service = FakeAuthService()
    app = build_test_app(monkeypatch, service=service, current_user=current_user)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/auth/become-admin")

    data = response.json()

    assert response.status_code == 200
    assert data["access_token"] == f"token:{current_user.id}"
    assert data["user"]["role"] == "admin"
    assert service.promoted_input_user is current_user


@pytest.mark.asyncio
async def test_get_admin_analytics_returns_counters(monkeypatch):
    """Verifies that admin analytics returns users, chats, and messages counters."""

    app = build_test_app(monkeypatch)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/auth/admin/analytics")

    assert response.status_code == 200
    assert response.json() == {
        "users_count": 2,
        "chats_count": 3,
        "messages_count": 5,
    }


@pytest.mark.asyncio
async def test_get_admin_analytics_requires_admin(monkeypatch):
    """Verifies that admin analytics returns HTTP 403 for non-admin users."""

    app = build_test_app(monkeypatch, admin_allowed=False)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/auth/admin/analytics")

    assert response.status_code == 403
    assert response.json()["detail"] == "Доступ только для администраторов"


@pytest.mark.asyncio
async def test_get_admin_analytics_uses_response_model(monkeypatch):
    """Verifies that admin analytics response fields have the expected types."""

    app = build_test_app(monkeypatch)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/auth/admin/analytics")

    data = response.json()

    assert response.status_code == 200
    assert isinstance(data["users_count"], int)
    assert isinstance(data["chats_count"], int)
    assert isinstance(data["messages_count"], int)