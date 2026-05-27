from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    """DTO запроса регистрации пользователя."""

    name: str = Field(min_length=2, max_length=255)
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=6, max_length=255)
    password_confirm: str = Field(min_length=6, max_length=255)


class LoginRequest(BaseModel):
    """DTO запроса авторизации пользователя."""

    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=6, max_length=255)


class AuthUserResponse(BaseModel):
    """DTO ответа с данными авторизованного пользователя."""

    id: UUID
    name: str
    email: str
    role: str
    created_at: datetime


class TokenResponse(BaseModel):
    """DTO ответа с JWT-токеном и пользователем."""

    access_token: str
    token_type: str = "bearer"
    user: AuthUserResponse


class AdminAnalyticsResponse(BaseModel):
    """DTO ответа административной аналитики auth-модуля."""

    users_count: int
    chats_count: int
    messages_count: int
