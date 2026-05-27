from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import get_db
from backend.db.users import User
from backend.modules.auth.dependencies import get_current_user
from backend.modules.chats.schema import (
    ChatCreateRequest,
    ChatResponse,
    ChatUpdateRequest,
)
from backend.modules.chats.service import chats_service

router = APIRouter(
    prefix="/api/chats",
    tags=["chats"],
)


@router.get(
    "",
    response_model=list[ChatResponse],
)
async def get_chats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Возвращает список чатов текущего пользователя."""
    return await chats_service.list_chats_response(
        db=db,
        current_user=current_user,
    )


@router.post(
    "",
    response_model=ChatResponse,
)
async def post_chat(
    payload: ChatCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Создаёт новый чат текущего пользователя."""
    return await chats_service.create_chat_response(
        db=db,
        payload=payload,
        current_user=current_user,
    )


@router.patch(
    "/{chat_id}",
    response_model=ChatResponse,
)
async def patch_chat(
    chat_id: UUID,
    payload: ChatUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Изменяет название чата текущего пользователя."""
    updated = await chats_service.update_chat_response(
        db=db,
        chat_id=chat_id,
        payload=payload,
        current_user=current_user,
    )

    if not updated:
        raise HTTPException(
            status_code=404,
            detail="Чат не найден",
        )

    return updated


@router.delete("/{chat_id}")
async def delete_chat(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Удаляет чат текущего пользователя."""
    result = await chats_service.delete_chat_response(
        db=db,
        chat_id=chat_id,
        current_user=current_user,
    )

    if result["status"] == "not_found":
        raise HTTPException(
            status_code=404,
            detail="Чат не найден",
        )

    return result
