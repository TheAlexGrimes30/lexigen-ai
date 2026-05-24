from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.schemas.chat import ChatCreateRequest, ChatResponse
from backend.modules.chats.service import chats_service


class ChatsController:
    async def list_chats(self, db: AsyncSession) -> list[ChatResponse]:
        chats = await chats_service.list_chats(db)
        return [ChatResponse.model_validate(chat) for chat in chats]

    async def create_chat(self, db: AsyncSession, payload: ChatCreateRequest) -> ChatResponse:
        chat = await chats_service.create_chat(db, payload.title)
        return ChatResponse.model_validate(chat)


chats_controller = ChatsController()
