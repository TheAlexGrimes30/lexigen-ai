import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException

from backend.app.config import settings
from backend.db.users import User
from backend.modules.auth.interfaces import BasePasswordHasher, BaseTokenManager


class Sha256PasswordHasher(BasePasswordHasher):
    """Хеширует и проверяет пароли через SHA-256."""

    def hash_password(
        self,
        password: str,
    ) -> str:
        """Возвращает SHA-256 хеш пароля."""
        return hashlib.sha256(
            password.encode("utf-8")
        ).hexdigest()

    def verify_password(
        self,
        raw_password: str,
        hashed_password: str,
    ) -> bool:
        """Проверяет пароль через безопасное сравнение хешей."""
        return hmac.compare_digest(
            self.hash_password(raw_password),
            hashed_password,
        )


class JwtTokenManager(BaseTokenManager):
    """Создаёт и проверяет JWT-токены без внешних зависимостей."""

    @staticmethod
    def _b64url_encode(
        data: bytes,
    ) -> str:
        """Кодирует байты в base64url без padding."""
        return base64.urlsafe_b64encode(
            data
        ).rstrip(b"=").decode("utf-8")

    @staticmethod
    def _b64url_decode(
        data: str,
    ) -> bytes:
        """Декодирует base64url строку с восстановлением padding."""
        padding = "=" * (-len(data) % 4)

        return base64.urlsafe_b64decode(
            data + padding
        )

    def create_access_token(
        self,
        user: User,
    ) -> str:
        """Создаёт подписанный JWT access token."""
        header = {
            "alg": "HS256",
            "typ": "JWT",
        }

        now = datetime.now(timezone.utc)
        exp = now + timedelta(
            minutes=settings.JWT_EXPIRES_MINUTES
        )

        payload = {
            "sub": str(user.id),
            "email": user.email,
            "role": user.role.value,
            "iat": int(now.timestamp()),
            "exp": int(exp.timestamp()),
        }

        header_b64 = self._b64url_encode(
            json.dumps(
                header,
                separators=(",", ":"),
            ).encode("utf-8")
        )

        payload_b64 = self._b64url_encode(
            json.dumps(
                payload,
                separators=(",", ":"),
            ).encode("utf-8")
        )

        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")

        signature = hmac.new(
            settings.JWT_SECRET.encode("utf-8"),
            signing_input,
            hashlib.sha256,
        ).digest()

        signature_b64 = self._b64url_encode(signature)

        return f"{header_b64}.{payload_b64}.{signature_b64}"

    def decode_access_token(
        self,
        token: str,
    ) -> dict:
        """Проверяет подпись JWT и возвращает payload."""
        try:
            header_b64, payload_b64, signature_b64 = token.split(".")
        except ValueError as exc:
            raise HTTPException(
                status_code=401,
                detail="Невалидный токен",
            ) from exc

        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")

        expected_signature = hmac.new(
            settings.JWT_SECRET.encode("utf-8"),
            signing_input,
            hashlib.sha256,
        ).digest()

        try:
            provided_signature = self._b64url_decode(signature_b64)
        except Exception as exc:
            raise HTTPException(
                status_code=401,
                detail="Невалидный токен",
            ) from exc

        if not hmac.compare_digest(
            expected_signature,
            provided_signature,
        ):
            raise HTTPException(
                status_code=401,
                detail="Невалидная подпись токена",
            )

        try:
            payload = json.loads(
                self._b64url_decode(payload_b64).decode("utf-8")
            )
        except Exception as exc:
            raise HTTPException(
                status_code=401,
                detail="Невалидный payload токена",
            ) from exc

        if int(datetime.now(timezone.utc).timestamp()) >= payload.get("exp", 0):
            raise HTTPException(
                status_code=401,
                detail="Токен истёк",
            )

        return payload
