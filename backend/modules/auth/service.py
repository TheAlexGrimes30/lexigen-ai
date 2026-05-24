import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import settings
from backend.db.enums import UserRole
from backend.db.users import User
from backend.modules.auth.schema import AuthUserResponse


class AuthService:
    @staticmethod
    def _b64url_encode(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")

    @staticmethod
    def _b64url_decode(data: str) -> bytes:
        padding = "=" * (-len(data) % 4)
        return base64.urlsafe_b64decode(data + padding)

    def hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def verify_password(self, raw_password: str, hashed_password: str) -> bool:
        return hmac.compare_digest(self.hash_password(raw_password), hashed_password)

    def create_access_token(self, user: User) -> str:
        header = {"alg": "HS256", "typ": "JWT"}
        now = datetime.now(timezone.utc)
        exp = now + timedelta(minutes=settings.JWT_EXPIRES_MINUTES)
        payload = {
            "sub": str(user.id),
            "email": user.email,
            "role": user.role.value,
            "iat": int(now.timestamp()),
            "exp": int(exp.timestamp()),
        }

        header_b64 = self._b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        payload_b64 = self._b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")

        signature = hmac.new(
            settings.JWT_SECRET.encode("utf-8"),
            signing_input,
            hashlib.sha256,
        ).digest()
        signature_b64 = self._b64url_encode(signature)
        return f"{header_b64}.{payload_b64}.{signature_b64}"

    def decode_access_token(self, token: str) -> dict:
        try:
            header_b64, payload_b64, signature_b64 = token.split(".")
        except ValueError as exc:
            raise HTTPException(status_code=401, detail="Невалидный токен") from exc

        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        expected_signature = hmac.new(
            settings.JWT_SECRET.encode("utf-8"),
            signing_input,
            hashlib.sha256,
        ).digest()

        try:
            provided_signature = self._b64url_decode(signature_b64)
        except Exception as exc:
            raise HTTPException(status_code=401, detail="Невалидный токен") from exc

        if not hmac.compare_digest(expected_signature, provided_signature):
            raise HTTPException(status_code=401, detail="Невалидная подпись токена")

        try:
            payload = json.loads(self._b64url_decode(payload_b64).decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=401, detail="Невалидный payload токена") from exc

        if int(datetime.now(timezone.utc).timestamp()) >= payload.get("exp", 0):
            raise HTTPException(status_code=401, detail="Токен истёк")

        return payload

    async def get_user_by_email(self, db: AsyncSession, email: str) -> User | None:
        stmt = select(User).where(User.email == email.lower())
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_user_by_id(self, db: AsyncSession, user_id: UUID) -> User | None:
        stmt = select(User).where(User.id == user_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def register(self, db: AsyncSession, name: str, email: str, password: str) -> User:
        normalized_email = email.lower().strip()
        existing = await self.get_user_by_email(db, normalized_email)
        if existing:
            raise HTTPException(status_code=409, detail="Пользователь с таким email уже существует")

        user = User(
            name=name.strip(),
            email=normalized_email,
            password_hash=self.hash_password(password),
            role=UserRole.admin if normalized_email == settings.ADMIN_EMAIL.lower() else UserRole.user,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user

    async def login(self, db: AsyncSession, email: str, password: str) -> User:
        normalized_email = email.lower().strip()
        user = await self.get_user_by_email(db, normalized_email)
        if not user or not self.verify_password(password, user.password_hash):
            raise HTTPException(status_code=401, detail="Неверный email или пароль")

        user.last_login_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(user)
        return user

    def to_auth_user(self, user: User) -> AuthUserResponse:
        return AuthUserResponse(
            id=user.id,
            name=user.name,
            email=user.email,
            role=user.role.value,
            created_at=user.created_at,
        )


auth_service = AuthService()
