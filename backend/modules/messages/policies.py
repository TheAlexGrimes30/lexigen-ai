from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.enums import UserRole
from backend.db.users import User
from backend.modules.messages.interfaces import (
    BaseDocumentAnalysisPolicy,
    BaseMessagesRepository,
)
from backend.modules.subscriptions.service import subscriptions_service


class DocumentAnalysisPolicy(BaseDocumentAnalysisPolicy):
    """Политика ограничения анализа документов по роли и подписке."""

    def __init__(
        self,
        repository: BaseMessagesRepository,
    ) -> None:
        """Инициализирует политику анализа документов."""
        self.repository = repository

    async def ensure_allowed(
        self,
        db: AsyncSession,
        current_user: User,
    ) -> None:
        """Проверяет право пользователя анализировать документ."""
        if current_user.role == UserRole.admin:
            return

        has_subscription = await subscriptions_service.user_has_paid_subscription(
            db,
            current_user.id,
        )

        if has_subscription:
            return

        used_uploads = await self.repository.count_user_documents(
            db=db,
            user_id=current_user.id,
        )

        if used_uploads >= 1:
            raise HTTPException(
                status_code=403,
                detail=(
                    "Без подписки доступна только одна загрузка документа. "
                    "Оформите Basic, Pro или Enterprise в личном кабинете."
                ),
            )
