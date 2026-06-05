import uuid
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from backend.db import User, UserRole
from backend.modules.auth.dependencies import get_current_user, get_admin_user


class FakeAuthService:
    """Fake auth service for dependency unit tests."""

    def __init__(self, payload: dict | None = None, user: User | None = None) -> None:
        self.payload = payload or {}
        self.user = user
        self.decoded_token: str | None = None
        self.requested_user_id: uuid.UUID | None = None

    def decode_access_token(self, token: str) -> dict:
        """Returns the configured decoded payload."""

        self.decoded_token = token
        return self.payload

    async def get_user_by_id(self, db: Any, user_id: uuid.UUID) -> User | None:
        """Returns the configured user lookup result."""

        self.requested_user_id = user_id
        return self.user

def make_user(role: UserRole = UserRole.user) -> User:
    """Builds a user model for dependency unit tests."""

    user = User(
        id=uuid.uuid4(),
        name="Test User",
        email="user@test.com",
        password_hash="hash",
        role=role,
    )
    user.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return user

@pytest.mark.asyncio
async def test_get_current_user_returns_user_for_valid_credentials(monkeypatch):
    """Verifies that a valid Bearer token resolves the current user."""

    user = make_user()
    fake_service = FakeAuthService(payload={"sub": str(user.id)}, user=user)
    monkeypatch.setattr("backend.modules.auth.dependencies.auth_service", fake_service)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid-token")

    result = await get_current_user(credentials=credentials, db=None)

    assert result is user
    assert fake_service.decoded_token == "valid-token"
    assert fake_service.requested_user_id == user.id

@pytest.mark.asyncio
async def test_get_current_user_rejects_missing_credentials():
    """Verifies that missing Authorization credentials return HTTP 401."""

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(credentials=None, db=None)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Требуется авторизация"

@pytest.mark.asyncio
async def test_get_current_user_rejects_empty_token():
    """Verifies that an empty Bearer token returns HTTP 401."""

    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="")

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(credentials=credentials, db=None)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Требуется авторизация"

@pytest.mark.asyncio
async def test_get_current_user_rejects_token_without_subject(monkeypatch):
    """Verifies that a decoded token without sub returns HTTP 401."""

    fake_service = FakeAuthService(payload={})
    monkeypatch.setattr("backend.modules.auth.dependencies.auth_service", fake_service)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="token")

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(credentials=credentials, db=None)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Некорректный токен"

@pytest.mark.asyncio
async def test_get_current_user_rejects_invalid_subject_uuid(monkeypatch):
    """Verifies that a non-UUID token subject returns HTTP 401."""

    fake_service = FakeAuthService(payload={"sub": "not-a-uuid"})
    monkeypatch.setattr("backend.modules.auth.dependencies.auth_service", fake_service)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="token")

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(credentials=credentials, db=None)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Некорректный токен"

@pytest.mark.asyncio
async def test_get_current_user_rejects_missing_user(monkeypatch):
    """Verifies that a valid token for a missing user returns HTTP 401."""

    user_id = uuid.uuid4()
    fake_service = FakeAuthService(payload={"sub": str(user_id)}, user=None)
    monkeypatch.setattr("backend.modules.auth.dependencies.auth_service", fake_service)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="token")

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(credentials=credentials, db=None)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Пользователь не найден"

@pytest.mark.asyncio
async def test_get_admin_user_returns_admin_user():
    """Verifies that admin users pass the admin dependency."""

    admin = make_user(role=UserRole.admin)
    result = await get_admin_user(current_user=admin)
    assert result is admin

@pytest.mark.asyncio
async def test_get_admin_user_rejects_regular_user():
    """Verifies that regular users receive HTTP 403 from the admin dependency."""

    user = make_user(role=UserRole.user)

    with pytest.raises(HTTPException) as exc_info:
        await get_admin_user(current_user=user)

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Доступ только для администраторов"