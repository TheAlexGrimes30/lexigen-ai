from uuid import UUID

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import get_db
from backend.db.users import User
from backend.modules.auth.service import auth_service

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Требуется авторизация")

    payload = auth_service.decode_access_token(credentials.credentials)
    subject = payload.get("sub")
    if not subject:
        raise HTTPException(status_code=401, detail="Некорректный токен")

    try:
        user_id = UUID(subject)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Некорректный токен") from exc

    user = await auth_service.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="Пользователь не найден")

    return user
