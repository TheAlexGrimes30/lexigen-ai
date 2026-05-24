from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import get_db
from backend.db.users import User
from backend.modules.auth.dependencies import get_current_user
from backend.modules.auth.schema import LoginRequest, RegisterRequest, TokenResponse, AuthUserResponse
from backend.modules.auth.service import auth_service

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
async def register(payload: RegisterRequest, db: AsyncSession = Depends(get_db)):
    user = await auth_service.register(db, payload.name, payload.email, payload.password)
    token = auth_service.create_access_token(user)
    return TokenResponse(access_token=token, user=auth_service.to_auth_user(user))


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await auth_service.login(db, payload.email, payload.password)
    token = auth_service.create_access_token(user)
    return TokenResponse(access_token=token, user=auth_service.to_auth_user(user))


@router.get("/me", response_model=AuthUserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return auth_service.to_auth_user(current_user)
