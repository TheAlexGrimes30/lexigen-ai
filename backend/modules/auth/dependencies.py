from uuid import UUID

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.logger_config import get_logger
from backend.db.database import get_db
from backend.db.enums import UserRole
from backend.db.users import User
from backend.modules.auth.service import auth_service

logger = get_logger(__name__)
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Возвращает текущего пользователя по JWT-токену."""

    if not credentials or not credentials.credentials:
        logger.warning("Unauthorized request without bearer token")

        raise HTTPException(
            status_code=401,
            detail="Требуется авторизация",
        )

    payload = auth_service.decode_access_token(
        credentials.credentials
    )

    subject = payload.get("sub")

    if not subject:
        logger.warning("JWT token does not contain subject")

        raise HTTPException(
            status_code=401,
            detail="Некорректный токен",
        )

    try:
        user_id = UUID(subject)
    except ValueError as exc:
        logger.warning("JWT token contains invalid user id")

        raise HTTPException(
            status_code=401,
            detail="Некорректный токен",
        ) from exc

    user = await auth_service.get_user_by_id(
        db,
        user_id,
    )

    if not user:
        logger.warning(
            "Authenticated user not found: user_id=%s",
            user_id,
        )

        raise HTTPException(
            status_code=401,
            detail="Пользователь не найден",
        )

    logger.info(
        "User authenticated successfully: user_id=%s",
        user.id,
    )

    return user


async def get_admin_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Возвращает текущего пользователя, если он администратор."""

    if current_user.role != UserRole.admin:
        logger.warning(
            "Admin access denied: user_id=%s",
            current_user.id,
        )

        raise HTTPException(
            status_code=403,
            detail="Доступ только для администраторов",
        )

    logger.info(
        "Admin access granted: user_id=%s",
        current_user.id,
    )

    return current_user
