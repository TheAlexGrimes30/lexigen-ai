from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.logger_config import get_logger
from backend.db.enums import UserRole
from backend.db.users import User
from backend.modules.messages.interfaces import (
    BaseDocumentAnalysisPolicy,
    BaseMessagesRepository,
)
from backend.modules.subscriptions.service import subscriptions_service

logger = get_logger(__name__)


class DocumentAnalysisPolicy(BaseDocumentAnalysisPolicy):
    """Политика ограничения анализа документов по роли и подписке."""

    def __init__(
        self,
        repository: BaseMessagesRepository,
    ):
        self.repository = repository

    async def ensure_allowed(
        self,
        db: AsyncSession,
        current_user: User,
    ) -> None:
        """Проверяет право пользователя анализировать документ."""

        logger.info(
            "Checking document analysis permission: user_id=%s, role=%s",
            current_user.id,
            current_user.role.value,
        )

        if current_user.role == UserRole.admin:
            logger.info(
                "Document analysis allowed for admin: user_id=%s",
                current_user.id,
            )
            return

        has_subscription = await subscriptions_service.user_has_paid_subscription(
            db,
            current_user.id,
        )

        if has_subscription:
            logger.info(
                "Document analysis allowed by paid subscription: user_id=%s",
                current_user.id,
            )
            return

        used_uploads = await self.repository.count_user_documents(
            db=db,
            user_id=current_user.id,
        )

        if used_uploads >= 1:
            logger.warning(
                "Document analysis denied: free upload limit reached: user_id=%s, used_uploads=%s",
                current_user.id,
                used_uploads,
            )

            raise HTTPException(
                status_code=403,
                detail=(
                    "Без подписки доступна только одна загрузка документа. "
                    "Оформите Basic, Pro или Enterprise в личном кабинете."
                ),
            )

        logger.info(
            "Document analysis allowed by free upload limit: user_id=%s, used_uploads=%s",
            current_user.id,
            used_uploads,
        )
