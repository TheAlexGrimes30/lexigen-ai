from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.chats import Chat
from backend.db.users import User

DEMO_USER_ID = UUID("11111111-1111-1111-1111-111111111111")


class ChatsService:
    async def ensure_demo_user(self, db: AsyncSession) -> User:
        stmt = select(User).where(User.id == DEMO_USER_ID)
        existing_user = (await db.execute(stmt)).scalar_one_or_none()
        if existing_user:
            return existing_user

        demo_user = User(
            id=DEMO_USER_ID,
            name="Demo User",
            email="demo@lexigen.local",
            password_hash="demo_password_hash",
        )
        db.add(demo_user)
        await db.commit()
        await db.refresh(demo_user)
        return demo_user

    async def list_chats(self, db: AsyncSession) -> list[Chat]:
        user = await self.ensure_demo_user(db)
        stmt = select(Chat).where(Chat.user_id == user.id).order_by(Chat.created_at.desc())
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def create_chat(self, db: AsyncSession, title: str) -> Chat:
        user = await self.ensure_demo_user(db)
        chat = Chat(title=title, user_id=user.id)
        db.add(chat)
        await db.commit()
        await db.refresh(chat)
        return chat

    async def get_chat(self, db: AsyncSession, chat_id: UUID) -> Chat | None:
        stmt = select(Chat).where(Chat.id == chat_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()


chats_service = ChatsService()
