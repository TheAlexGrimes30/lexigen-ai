import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException

from backend.db import User, UserRole
from backend.modules.auth.security import JwtTokenManager, Sha256PasswordHasher


def make_user() -> User:
    """Builds a user model for security unit tests."""

    user = User(
        id=uuid.uuid4(),
        name="Test User",
        email="user@test.com",
        password_hash="hash",
        role=UserRole.user,
    )
    user.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    return user


def tamper_payload(token: str) -> str:
    """Returns a JWT with a modified payload and the original signature."""

    header_b64, payload_b64, signature_b64 = token.split(".")
    payload = json.loads(
        JwtTokenManager._b64url_decode(payload_b64).decode("utf-8")
    )
    payload["email"] = "attacker@test.com"

    modified_payload_b64 = JwtTokenManager._b64url_encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    )

    return f"{header_b64}.{modified_payload_b64}.{signature_b64}"


def test_sha256_password_hasher_returns_deterministic_hash():
    """Verifies that equal passwords produce equal SHA-256 hashes."""

    hasher = Sha256PasswordHasher()

    assert hasher.hash_password("secret123") == hasher.hash_password("secret123")


def test_sha256_password_hasher_verifies_valid_password():
    """Verifies that the hasher accepts a matching raw password."""

    hasher = Sha256PasswordHasher()
    hashed = hasher.hash_password("secret123")

    assert hasher.verify_password("secret123", hashed) is True


def test_sha256_password_hasher_rejects_invalid_password():
    """Verifies that the hasher rejects a non-matching raw password."""

    hasher = Sha256PasswordHasher()
    hashed = hasher.hash_password("secret123")

    assert hasher.verify_password("wrong-password", hashed) is False


def test_b64url_encode_strips_padding():
    """Verifies that base64url encoding does not include padding characters."""

    encoded = JwtTokenManager._b64url_encode(b"test")

    assert "=" not in encoded
    assert JwtTokenManager._b64url_decode(encoded) == b"test"


def test_create_and_decode_access_token_returns_payload(monkeypatch):
    """Verifies that a created access token can be decoded into a valid payload."""

    monkeypatch.setattr(
        "backend.modules.auth.security.settings.JWT_SECRET",
        "test-secret",
    )
    monkeypatch.setattr(
        "backend.modules.auth.security.settings.JWT_EXPIRES_MINUTES",
        30,
    )

    manager = JwtTokenManager()
    user = make_user()

    token = manager.create_access_token(user)
    payload = manager.decode_access_token(token)

    assert payload["sub"] == str(user.id)
    assert payload["email"] == user.email
    assert payload["role"] == user.role.value
    assert payload["exp"] > int(time.time())


def test_decode_access_token_rejects_malformed_token():
    """Verifies that malformed JWT strings return HTTP 401."""

    manager = JwtTokenManager()

    with pytest.raises(HTTPException) as exc_info:
        manager.decode_access_token("not-a-jwt")

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Невалидный токен"


def test_decode_access_token_rejects_invalid_signature(monkeypatch):
    """Verifies that a token with a tampered payload is rejected."""

    monkeypatch.setattr(
        "backend.modules.auth.security.settings.JWT_SECRET",
        "test-secret",
    )

    manager = JwtTokenManager()
    token = manager.create_access_token(make_user())

    with pytest.raises(HTTPException) as exc_info:
        manager.decode_access_token(tamper_payload(token))

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Невалидная подпись токена"


def test_decode_access_token_rejects_invalid_signature_encoding():
    """Verifies that invalid signature data returns HTTP 401."""

    manager = JwtTokenManager()

    with pytest.raises(HTTPException) as exc_info:
        manager.decode_access_token("header.payload.***")

    assert exc_info.value.status_code == 401


def test_decode_access_token_rejects_invalid_payload(monkeypatch):
    """Verifies that a token with invalid payload JSON is rejected."""

    monkeypatch.setattr(
        "backend.modules.auth.security.settings.JWT_SECRET",
        "test-secret",
    )

    manager = JwtTokenManager()
    header = manager._b64url_encode(b'{"alg":"HS256","typ":"JWT"}')
    payload = manager._b64url_encode(b"not-json")
    signing_input = f"{header}.{payload}".encode("utf-8")

    signature = hmac.new(
        b"test-secret",
        signing_input,
        hashlib.sha256,
    ).digest()
    signature_b64 = manager._b64url_encode(signature)

    with pytest.raises(HTTPException) as exc_info:
        manager.decode_access_token(f"{header}.{payload}.{signature_b64}")

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Невалидный payload токена"


def test_decode_access_token_rejects_expired_token(monkeypatch):
    """Verifies that expired access tokens return HTTP 401."""

    monkeypatch.setattr(
        "backend.modules.auth.security.settings.JWT_SECRET",
        "test-secret",
    )
    monkeypatch.setattr(
        "backend.modules.auth.security.settings.JWT_EXPIRES_MINUTES",
        -1,
    )

    manager = JwtTokenManager()
    token = manager.create_access_token(make_user())

    with pytest.raises(HTTPException) as exc_info:
        manager.decode_access_token(token)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Токен истёк"