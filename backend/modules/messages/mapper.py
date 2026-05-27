from backend.db.messages import Message
from backend.modules.messages.schema import MessageResponse


class MessageMapper:
    """Маппер ORM-сообщений в DTO."""

    def to_response(
        self,
        message: Message,
    ) -> MessageResponse:
        """Преобразует ORM-сообщение в DTO."""
        return MessageResponse.model_validate(message)

    def to_response_list(
        self,
        messages: list[Message],
    ) -> list[MessageResponse]:
        """Преобразует список ORM-сообщений в список DTO."""
        return [
            self.to_response(message)
            for message in messages
        ]
