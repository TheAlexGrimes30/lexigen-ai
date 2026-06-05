import uuid

import pytest
from fastapi import HTTPException

from backend.db import User, UserRole
from backend.modules.messages.policies import DocumentAnalysisPolicy


class FakeMessagesRepository:
    """Fake repository for document analysis policy tests."""

    def __init__(self, document_count: int) -> None:
        """Stores document count returned by the fake repository."""

        self.document_count = document_count
        self.count_calls = []

    async def count_user_documents(self, db, user_id):
        """Returns a predefined document count."""

        self.count_calls.append(user_id)
        return self.document_count


def make_user(role: UserRole = UserRole.user) -> User:
    """Builds a user model for policy unit tests."""

    return User(
        id=uuid.uuid4(),
        name="Test User",
        email="user@test.com",
        password_hash="hash",
        role=role,
    )


@pytest.mark.asyncio
async def test_policy_allows_admin_without_subscription_check(monkeypatch):
    """Verifies that administrators are always allowed to analyze documents."""

    async def fail_subscription_check(db, user_id):
        """Fails if the subscription check is unexpectedly called."""

        raise AssertionError("Subscription check should not be called")

    monkeypatch.setattr(
        "backend.modules.messages.policies.subscriptions_service.user_has_paid_subscription",
        fail_subscription_check,
    )

    repository = FakeMessagesRepository(document_count=99)
    policy = DocumentAnalysisPolicy(repository=repository)

    await policy.ensure_allowed(db=None, current_user=make_user(UserRole.admin))

    assert repository.count_calls == []


@pytest.mark.asyncio
async def test_policy_allows_user_with_paid_subscription(monkeypatch):
    """Verifies that paid subscribers can analyze documents."""

    async def has_subscription(db, user_id):
        """Returns that the user has a paid subscription."""

        return True

    monkeypatch.setattr(
        "backend.modules.messages.policies.subscriptions_service.user_has_paid_subscription",
        has_subscription,
    )

    repository = FakeMessagesRepository(document_count=99)
    policy = DocumentAnalysisPolicy(repository=repository)

    await policy.ensure_allowed(db=None, current_user=make_user())

    assert repository.count_calls == []


@pytest.mark.asyncio
async def test_policy_allows_first_free_upload(monkeypatch):
    """Verifies that a user without subscription can upload one document."""

    async def no_subscription(db, user_id):
        """Returns that the user has no paid subscription."""

        return False

    monkeypatch.setattr(
        "backend.modules.messages.policies.subscriptions_service.user_has_paid_subscription",
        no_subscription,
    )

    user = make_user()
    repository = FakeMessagesRepository(document_count=0)
    policy = DocumentAnalysisPolicy(repository=repository)

    await policy.ensure_allowed(db=None, current_user=user)

    assert repository.count_calls == [user.id]


@pytest.mark.asyncio
async def test_policy_rejects_second_free_upload(monkeypatch):
    """Verifies that a user without subscription cannot upload twice."""

    async def no_subscription(db, user_id):
        """Returns that the user has no paid subscription."""

        return False

    monkeypatch.setattr(
        "backend.modules.messages.policies.subscriptions_service.user_has_paid_subscription",
        no_subscription,
    )

    policy = DocumentAnalysisPolicy(repository=FakeMessagesRepository(document_count=1))

    with pytest.raises(HTTPException) as exc_info:
        await policy.ensure_allowed(db=None, current_user=make_user())

    assert exc_info.value.status_code == 403
    assert "Без подписки доступна только одна загрузка документа" in exc_info.value.detail
