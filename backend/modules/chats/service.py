from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.chats import Chat
from backend.db.users import User
from backend.modules.chats.schema import ChatCreateRequest, ChatResponse


class ChatsService:
    async def list_chats(self, db: AsyncSession, current_user: User) -> list[Chat]:
        stmt = select(Chat).where(Chat.user_id == current_user.id).order_by(Chat.created_at.desc())
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def create_chat(self, db: AsyncSession, title: str, current_user: User) -> Chat:
        chat = Chat(title=title, user_id=current_user.id)
        db.add(chat)
        await db.commit()
        await db.refresh(chat)
        return chat

    async def get_chat(self, db: AsyncSession, chat_id: UUID, current_user: User) -> Chat | None:
        stmt = select(Chat).where(Chat.id == chat_id, Chat.user_id == current_user.id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def delete_chat(self, db: AsyncSession, chat_id: UUID, current_user: User) -> bool:
        chat = await self.get_chat(db, chat_id, current_user)
        if not chat:
            return False

        await db.delete(chat)
        await db.commit()
        return True

    async def list_chats_response(self, db: AsyncSession, current_user: User) -> list[ChatResponse]:
        chats = await self.list_chats(db, current_user)
        return [ChatResponse.model_validate(chat) for chat in chats]

    async def create_chat_response(self, db: AsyncSession, payload: ChatCreateRequest, current_user: User) -> ChatResponse:
        chat = await self.create_chat(db, payload.title, current_user)
        return ChatResponse.model_validate(chat)

    async def delete_chat_response(self, db: AsyncSession, chat_id: UUID, current_user: User) -> dict[str, str]:
        deleted = await self.delete_chat(db, chat_id, current_user)
        if not deleted:
            return {"status": "not_found"}
        return {"status": "deleted"}


chats_service = ChatsService()
