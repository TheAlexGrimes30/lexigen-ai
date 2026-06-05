import uuid
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi import HTTPException

from backend.db.enums import UserRole
from backend.db.users import User
from backend.modules.auth.schema import AdminAnalyticsResponse, AuthUserResponse
from backend.modules.auth.service import AuthService


class FakeAuthRepository:
    """Fake repository used to isolate AuthService unit tests from the database."""

    def __init__(self) -> None:
        self.users_by_email: dict[str, User] = {}
        self.users_by_id: dict[uuid.UUID, User] = {}
        self.added_user: User | None = None
        self.saved_user: User | None = None
        self.users_count = 0
        self.chats_count = 0
        self.messages_count = 0

    async def get_user_by_email(self, db: Any, email: str) -> User | None:
        """Returns a user by normalized email from memory."""

        return self.users_by_email.get(email.lower())

    async def get_user_by_id(self, db: Any, user_id: uuid.UUID) -> User | None:
        """Returns a user by UUID from memory."""

        return self.users_by_id.get(user_id)

    async def add_user(self, db: Any, user: User) -> User:
        """Stores a newly registered user in memory."""

        if user.id is None:
            user.id = uuid.uuid4()
        self.added_user = user
        self.users_by_email[user.email] = user
        self.users_by_id[user.id] = user
        return user

    async def save_user(self, db: Any, user: User) -> User:
        """Stores the latest user state in memory."""

        self.saved_user = user
        self.users_by_email[user.email] = user
        self.users_by_id[user.id] = user
        return user

    async def count_users(self, db: Any) -> int:
        """Returns the configured users counter."""

        return self.users_count

    async def count_chats(self, db: Any) -> int:
        """Returns the configured chats counter."""

        return self.chats_count

    async def count_messages(self, db: Any) -> int:
        """Returns the configured messages counter."""

        return self.messages_count


class FakePasswordHasher:
    """Fake password hasher with deterministic hash values."""

    def hash_password(self, password: str) -> str:
        """Returns a deterministic fake hash for a raw password."""

        return f"hashed:{password}"

    def verify_password(self, raw_password: str, hashed_password: str) -> bool:
        """Checks whether the fake hash matches the raw password."""

        return hashed_password == self.hash_password(raw_password)


class FakeTokenManager:
    """Fake token manager for deterministic token tests."""

    def create_access_token(self, user: User) -> str:
        """Returns a deterministic fake access token."""

        return f"token:{user.id}"

    def decode_access_token(self, token: str) -> dict:
        """Returns a deterministic decoded token payload."""

        return {"sub": token.removeprefix("token:")}


def make_user(
    *,
    user_id: uuid.UUID | None = None,
    name: str = "Test User",
    email: str = "user@test.com",
    password_hash: str = "hashed:secret123",
    role: UserRole = UserRole.user,
) -> User:
    """Builds a user model for service unit tests."""

    user = User(
        id=user_id or uuid.uuid4(),
        name=name,
        email=email,
        password_hash=password_hash,
        role=role,
    )
    user.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return user


@pytest.fixture
def repository() -> FakeAuthRepository:
    """Provides a fake repository for each test."""
    return FakeAuthRepository()


@pytest.fixture
def service(repository: FakeAuthRepository) -> AuthService:
    """Provides an AuthService wired with fake dependencies."""

    return AuthService(
        repository=repository,
        password_hasher=FakePasswordHasher(),
        token_manager=FakeTokenManager(),
    )


def test_hash_password_delegates_to_password_hasher(service: AuthService):
    """Verifies that password hashing is delegated to the configured hasher."""

    assert service.hash_password("secret123") == "hashed:secret123"


def test_verify_password_returns_true_for_matching_hash(service: AuthService):
    """Verifies that password verification succeeds for matching credentials."""

    assert service.verify_password("secret123", "hashed:secret123") is True


def test_verify_password_returns_false_for_wrong_hash(service: AuthService):
    """Verifies that password verification fails for invalid credentials."""

    assert service.verify_password("wrong", "hashed:secret123") is False


def test_create_access_token_delegates_to_token_manager(service: AuthService):
    """Verifies that access token creation is delegated to the token manager."""

    user = make_user()

    assert service.create_access_token(user) == f"token:{user.id}"


def test_decode_access_token_delegates_to_token_manager(service: AuthService):
    """Verifies that access token decoding is delegated to the token manager."""

    user_id = uuid.uuid4()

    assert service.decode_access_token(f"token:{user_id}") == {"sub": str(user_id)}


@pytest.mark.asyncio
async def test_get_user_by_email_reads_from_repository(service: AuthService, repository: FakeAuthRepository):
    """Verifies that users can be retrieved by normalized email."""

    user = make_user(email="user@test.com")
    repository.users_by_email[user.email] = user

    result = await service.get_user_by_email(db=None, email="user@test.com")

    assert result is user


@pytest.mark.asyncio
async def test_get_user_by_id_reads_from_repository(service: AuthService, repository: FakeAuthRepository):
    """Verifies that users can be retrieved by UUID."""

    user = make_user()
    repository.users_by_id[user.id] = user

    result = await service.get_user_by_id(db=None, user_id=user.id)

    assert result is user


@pytest.mark.asyncio
async def test_register_creates_regular_user_with_normalized_email(service: AuthService, repository: FakeAuthRepository):
    """Verifies that registration normalizes email, strips name, and creates a regular user."""

    result = await service.register(
        db=None,
        name="  John Doe  ",
        email="  USER@TEST.COM  ",
        password="secret123",
    )

    assert result is repository.added_user
    assert result.name == "John Doe"
    assert result.email == "user@test.com"
    assert result.password_hash == "hashed:secret123"
    assert result.role == UserRole.user


@pytest.mark.asyncio
async def test_register_raises_conflict_when_email_already_exists(service: AuthService, repository: FakeAuthRepository):
    """Verifies that duplicate email registration returns HTTP 409."""

    existing = make_user(email="user@test.com")
    repository.users_by_email[existing.email] = existing

    with pytest.raises(HTTPException) as exc_info:
        await service.register(
            db=None,
            name="John Doe",
            email="USER@TEST.COM",
            password="secret123",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Пользователь с таким email уже существует"


@pytest.mark.asyncio
async def test_register_creates_admin_when_email_matches_settings(service: AuthService, monkeypatch):
    """Verifies that the configured admin email receives the admin role on registration."""

    monkeypatch.setattr("backend.modules.auth.service.settings.ADMIN_EMAIL", "admin@test.com")

    result = await service.register(
        db=None,
        name="Admin",
        email="ADMIN@TEST.COM",
        password="secret123",
    )

    assert result.role == UserRole.admin


@pytest.mark.asyncio
async def test_login_saves_last_login_when_credentials_are_valid(service: AuthService, repository: FakeAuthRepository):
    """Verifies that login updates last_login_at and saves the user."""

    user = make_user(email="user@test.com", password_hash="hashed:secret123")
    repository.users_by_email[user.email] = user
    result = await service.login(db=None, email=" USER@TEST.COM ", password="secret123")

    assert result is user
    assert repository.saved_user is user
    assert user.last_login_at is not None


@pytest.mark.asyncio
async def test_login_raises_unauthorized_when_user_does_not_exist(service: AuthService):
    """Verifies that login fails with HTTP 401 when the email is unknown."""

    with pytest.raises(HTTPException) as exc_info:
        await service.login(db=None, email="missing@test.com", password="secret123")

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Неверный email или пароль"


@pytest.mark.asyncio
async def test_login_raises_unauthorized_when_password_is_invalid(service: AuthService, repository: FakeAuthRepository):
    """Verifies that login fails with HTTP 401 when the password is invalid."""

    user = make_user(email="user@test.com", password_hash="hashed:secret123")
    repository.users_by_email[user.email] = user

    with pytest.raises(HTTPException) as exc_info:
        await service.login(db=None, email="user@test.com", password="wrong-password")

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Неверный email или пароль"


@pytest.mark.asyncio
async def test_promote_to_admin_updates_regular_user(service: AuthService, repository: FakeAuthRepository):
    """Verifies that a regular user is promoted and saved."""

    user = make_user(role=UserRole.user)
    result = await service.promote_to_admin(db=None, user=user)

    assert result.role == UserRole.admin
    assert repository.saved_user is user


@pytest.mark.asyncio
async def test_promote_to_admin_does_not_save_existing_admin(service: AuthService, repository: FakeAuthRepository):
    """Verifies that an already-admin user is returned without another save."""

    user = make_user(role=UserRole.admin)
    result = await service.promote_to_admin(db=None, user=user)

    assert result is user
    assert repository.saved_user is None


@pytest.mark.asyncio
async def test_get_admin_analytics_returns_repository_counters(service: AuthService, repository: FakeAuthRepository):
    """Verifies that admin analytics are built from repository counters."""

    repository.users_count = 3
    repository.chats_count = 5
    repository.messages_count = 8

    result = await service.get_admin_analytics(db=None)

    assert isinstance(result, AdminAnalyticsResponse)
    assert result.users_count == 3
    assert result.chats_count == 5
    assert result.messages_count == 8


def test_to_auth_user_maps_user_model_to_response_schema(service: AuthService):
    """Verifies that a User model is converted into an AuthUserResponse DTO."""

    user = make_user(role=UserRole.admin)
    result = service.to_auth_user(user)

    assert isinstance(result, AuthUserResponse)
    assert result.id == user.id
    assert result.name == user.name
    assert result.email == user.email
    assert result.role == "admin"
    assert result.created_at == user.created_at
