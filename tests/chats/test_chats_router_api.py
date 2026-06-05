import uuid
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException, FastAPI
from httpx import AsyncClient, ASGITransport

from backend.db import UserRole, User
from backend.db.database import get_db
from backend.modules.auth.dependencies import get_current_user
from backend.modules.chats.router import router
from backend.modules.chats.schema import ChatResponse

CURRENT_USER_ID = uuid.uuid4()
CHAT_ID = uuid.uuid4()
MISSING_CHAT_ID = uuid.uuid4()
CREATED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeChatsService:
    """Fake chats service for router API tests."""

    def __init__(self) -> None:
        """Initializes call-tracking fields for API tests."""

        self.list_called = False
        self.created_payload_title: str | None = None
        self.updated_payload_title: str | None = None
        self.deleted_chat_id: uuid.UUID | None = None

    async def list_chats_response(self, db, current_user: User) -> list[ChatResponse]:
        """Returns predefined chats for the current user."""

        self.list_called = True
        return [
            ChatResponse(
                id=CHAT_ID,
                user_id=current_user.id,
                title="Existing chat",
                created_at=CREATED_AT,
            )
        ]

    async def create_chat_response(self, db, payload, current_user: User) -> ChatResponse:
        """Returns a chat DTO created from the request payload."""

        self.created_payload_title = payload.title
        return ChatResponse(
            id=CHAT_ID,
            user_id=current_user.id,
            title=payload.title,
            created_at=CREATED_AT,
        )

    async def update_chat_response(self, db, chat_id: uuid.UUID, payload, current_user: User) -> ChatResponse | None:
        """Returns an updated DTO unless the requested chat is missing."""

        self.updated_payload_title = payload.title
        if chat_id == MISSING_CHAT_ID:
            return None

        return ChatResponse(
            id=chat_id,
            user_id=current_user.id,
            title=payload.title,
            created_at=CREATED_AT,
        )

    async def delete_chat_response(self, db, chat_id: uuid.UUID, current_user: User) -> dict[str, str]:
        """Returns a deletion status for the requested chat."""

        self.deleted_chat_id = chat_id
        if chat_id == MISSING_CHAT_ID:
            return {"status": "not_found"}

        return {"status": "deleted"}


def make_current_user() -> User:
    """Builds an authenticated user for API tests."""

    user = User(
        id=CURRENT_USER_ID,
        name="Test User",
        email="user@test.com",
        password_hash="hash",
        role=UserRole.user,
    )
    user.created_at = CREATED_AT
    return user


def build_test_app(monkeypatch, service: FakeChatsService | None = None, authenticated: bool = True) -> FastAPI:
    """Builds a FastAPI app with overridden dependencies and fake service."""

    app = FastAPI()
    app.include_router(router)

    async def fake_get_db():
        """Returns a fake database dependency value."""

        yield object()

    async def fake_get_current_user():
        """Returns a fake user or raises HTTP 401 for unauthenticated requests."""

        if not authenticated:
            raise HTTPException(status_code=401, detail="Требуется авторизация")
        return make_current_user()

    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[get_current_user] = fake_get_current_user

    monkeypatch.setattr(
        "backend.modules.chats.router.chats_service",
        service or FakeChatsService(),
    )

    return app


@pytest.mark.asyncio
async def test_get_chats_returns_current_user_chats(monkeypatch):
    """Verifies that GET /api/chats returns current user chats."""

    app = build_test_app(monkeypatch)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/chats")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": str(CHAT_ID),
            "user_id": str(CURRENT_USER_ID),
            "title": "Existing chat",
            "created_at": CREATED_AT.isoformat().replace("+00:00", "Z"),
        }
    ]


@pytest.mark.asyncio
async def test_get_chats_requires_authentication(monkeypatch):
    """Verifies that GET /api/chats returns HTTP 401 without authentication."""

    app = build_test_app(monkeypatch, authenticated=False)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/chats")

    assert response.status_code == 401
    assert response.json()["detail"] == "Требуется авторизация"


@pytest.mark.asyncio
async def test_post_chat_creates_chat(monkeypatch):
    """Verifies that POST /api/chats creates a chat for the current user."""

    service = FakeChatsService()
    app = build_test_app(monkeypatch, service=service)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/api/chats", json={"title": "New chat"})

    data = response.json()

    assert response.status_code == 200
    assert data["id"] == str(CHAT_ID)
    assert data["user_id"] == str(CURRENT_USER_ID)
    assert data["title"] == "New chat"
    assert service.created_payload_title == "New chat"


@pytest.mark.asyncio
async def test_post_chat_rejects_empty_title(monkeypatch):
    """Verifies that POST /api/chats validates an empty title."""

    app = build_test_app(monkeypatch)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/api/chats", json={"title": ""})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_post_chat_rejects_too_long_title(monkeypatch):
    """Verifies that POST /api/chats validates the maximum title length."""

    app = build_test_app(monkeypatch)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/api/chats", json={"title": "x" * 256})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_patch_chat_updates_chat(monkeypatch):
    """Verifies that PATCH /api/chats/{chat_id} updates a chat title."""

    service = FakeChatsService()
    app = build_test_app(monkeypatch, service=service)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.patch(
            f"/api/chats/{CHAT_ID}",
            json={"title": "Updated chat"},
        )

    data = response.json()

    assert response.status_code == 200
    assert data["id"] == str(CHAT_ID)
    assert data["title"] == "Updated chat"
    assert service.updated_payload_title == "Updated chat"


@pytest.mark.asyncio
async def test_patch_chat_returns_404_when_chat_is_missing(monkeypatch):
    """Verifies that PATCH /api/chats/{chat_id} returns HTTP 404 for a missing chat."""

    app = build_test_app(monkeypatch)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.patch(
            f"/api/chats/{MISSING_CHAT_ID}",
            json={"title": "Updated chat"},
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "Чат не найден"


@pytest.mark.asyncio
async def test_patch_chat_rejects_invalid_uuid(monkeypatch):
    """Verifies that PATCH /api/chats/{chat_id} validates UUID path parameters."""

    app = build_test_app(monkeypatch)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.patch(
            "/api/chats/not-a-uuid",
            json={"title": "Updated chat"},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_patch_chat_rejects_empty_title(monkeypatch):
    """Verifies that PATCH /api/chats/{chat_id} validates an empty title."""

    app = build_test_app(monkeypatch)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.patch(
            f"/api/chats/{CHAT_ID}",
            json={"title": ""},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_delete_chat_returns_deleted_status(monkeypatch):
    """Verifies that DELETE /api/chats/{chat_id} deletes an existing chat."""

    service = FakeChatsService()
    app = build_test_app(monkeypatch, service=service)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.delete(f"/api/chats/{CHAT_ID}")

    assert response.status_code == 200
    assert response.json() == {"status": "deleted"}
    assert service.deleted_chat_id == CHAT_ID


@pytest.mark.asyncio
async def test_delete_chat_returns_404_when_chat_is_missing(monkeypatch):
    """Verifies that DELETE /api/chats/{chat_id} returns HTTP 404 for a missing chat."""

    app = build_test_app(monkeypatch)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.delete(f"/api/chats/{MISSING_CHAT_ID}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Чат не найден"


@pytest.mark.asyncio
async def test_delete_chat_rejects_invalid_uuid(monkeypatch):
    """Verifies that DELETE /api/chats/{chat_id} validates UUID path parameters."""

    app = build_test_app(monkeypatch)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.delete("/api/chats/not-a-uuid")

    assert response.status_code == 422
