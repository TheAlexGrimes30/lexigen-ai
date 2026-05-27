from backend.db.chats import Chat
from backend.modules.chats.schema import ChatResponse


class ChatMapper:
    """Маппер ORM-модели чата в DTO."""

    def to_response(
        self,
        chat: Chat,
    ) -> ChatResponse:
        """Преобразует ORM-чат в DTO ответа."""
        return ChatResponse.model_validate(chat)

    def to_response_list(
        self,
        chats: list[Chat],
    ) -> list[ChatResponse]:
        """Преобразует список ORM-чатов в список DTO."""
        return [
            self.to_response(chat)
            for chat in chats
        ]
