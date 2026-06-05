import asyncio

import pytest

from backend.modules.rag.service import RAGApplicationService, RAGStatus


class FakeRAG:
    """Fake RAG engine used to avoid loading real models in unit tests."""

    def __init__(self) -> None:
        """Initializes fake engine state."""
        self.index_built = False

    def build_and_index(self) -> None:
        """Marks the fake index as built."""
        self.index_built = True

    def ask(self, query: str) -> str:
        """Returns a deterministic fake answer."""
        return f"answer: {query}"

    def analyze_contract(self, contract_text: str) -> str:
        """Returns a deterministic fake contract analysis."""
        return f"analysis: {contract_text}"


async def run_to_thread_sync(func, *args, **kwargs):
    """Runs a synchronous callable instead of dispatching it to a thread."""
    return func(*args, **kwargs)


@pytest.mark.asyncio
async def test_startup_sets_loading_and_schedules_background_load(monkeypatch):
    """Verifies that startup switches the service to loading state."""

    service = RAGApplicationService()
    created_tasks = []

    def fake_create_task(coro):
        """Stores and closes the scheduled coroutine for assertion."""
        created_tasks.append(coro)
        coro.close()
        return object()

    monkeypatch.setattr(
        "backend.modules.rag.service.asyncio.create_task",
        fake_create_task,
    )

    await service.startup()

    assert service.status == RAGStatus.LOADING
    assert service.error is None
    assert len(created_tasks) == 1


@pytest.mark.asyncio
async def test_startup_does_not_schedule_when_already_loading(monkeypatch):
    """Verifies that startup is idempotent while RAG is loading."""

    service = RAGApplicationService()
    service._status = RAGStatus.LOADING

    def fail_create_task(coro):
        """Fails if startup tries to schedule another loading task."""
        coro.close()
        raise AssertionError("create_task must not be called")

    monkeypatch.setattr(
        "backend.modules.rag.service.asyncio.create_task",
        fail_create_task,
    )

    await service.startup()

    assert service.status == RAGStatus.LOADING

@pytest.mark.asyncio
async def test_startup_does_not_schedule_when_already_ready(monkeypatch):
    """Verifies that startup is idempotent when RAG is already ready."""

    service = RAGApplicationService()
    service._status = RAGStatus.READY
    service._rag = FakeRAG()

    def fail_create_task(coro):
        """Fails if startup tries to schedule another loading task."""
        coro.close()
        raise AssertionError("create_task must not be called")

    monkeypatch.setattr(
        "backend.modules.rag.service.asyncio.create_task",
        fail_create_task,
    )

    await service.startup()

    assert service.status == RAGStatus.READY

@pytest.mark.asyncio
async def test_load_sets_ready_when_rag_build_succeeds(monkeypatch):
    """Verifies that _load initializes RAG and marks the service as ready."""

    service = RAGApplicationService()
    service._status = RAGStatus.LOADING

    monkeypatch.setattr("backend.modules.rag.service.RAG", FakeRAG)
    monkeypatch.setattr(
        "backend.modules.rag.service.asyncio.to_thread",
        run_to_thread_sync,
    )

    await service._load()

    assert service.status == RAGStatus.READY
    assert service.error is None
    assert isinstance(service._rag, FakeRAG)
    assert service._rag.index_built is True

@pytest.mark.asyncio
async def test_load_sets_error_when_rag_build_fails(monkeypatch):
    """Verifies that _load stores an error when RAG initialization fails."""

    service = RAGApplicationService()
    service._status = RAGStatus.LOADING

    class BrokenRAG:
        """Fake RAG engine that fails during indexing."""

        def build_and_index(self):
            """Raises a deterministic loading error."""
            raise RuntimeError("index failed")

    monkeypatch.setattr("backend.modules.rag.service.RAG", BrokenRAG)
    monkeypatch.setattr(
        "backend.modules.rag.service.asyncio.to_thread",
        run_to_thread_sync,
    )

    await service._load()

    assert service.status == RAGStatus.ERROR
    assert service._rag is None
    assert service.error == "index failed"

@pytest.mark.asyncio
async def test_ask_returns_empty_query_message_without_ready_check():
    """Verifies that empty user queries return a user-facing message."""

    service = RAGApplicationService()

    result = await service.ask("   ")

    assert result == "Пустой запрос. Напишите вопрос, чтобы я смог найти ответ."

@pytest.mark.asyncio
async def test_ask_returns_rag_answer_when_ready(monkeypatch):
    """Verifies that ask delegates a non-empty query to the RAG engine."""

    service = RAGApplicationService()
    service._status = RAGStatus.READY
    service._rag = FakeRAG()

    monkeypatch.setattr(
        "backend.modules.rag.service.asyncio.to_thread",
        run_to_thread_sync,
    )

    result = await service.ask("  test query  ")

    assert result == "answer: test query"


@pytest.mark.asyncio
async def test_ask_wraps_rag_errors(monkeypatch):
    """Verifies that ask converts engine exceptions into RuntimeError."""

    service = RAGApplicationService()
    service._status = RAGStatus.READY

    class BrokenAskRAG:
        """Fake RAG engine that fails on ask."""

        def ask(self, query: str) -> str:
            """Raises a deterministic ask error."""
            raise ValueError("generation failed")

    service._rag = BrokenAskRAG()

    monkeypatch.setattr(
        "backend.modules.rag.service.asyncio.to_thread",
        run_to_thread_sync,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await service.ask("query")

    assert "Ошибка при генерации RAG-ответа: generation failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_analyze_contract_returns_empty_text_message_without_ready_check():
    """Verifies that empty contract text returns a user-facing message."""

    service = RAGApplicationService()

    result = await service.analyze_contract("   ")

    assert result == "Не удалось извлечь текст из документа."


@pytest.mark.asyncio
async def test_analyze_contract_returns_rag_analysis_when_ready(monkeypatch):
    """Verifies that contract analysis delegates to the RAG engine."""

    service = RAGApplicationService()
    service._status = RAGStatus.READY
    service._rag = FakeRAG()

    monkeypatch.setattr(
        "backend.modules.rag.service.asyncio.to_thread",
        run_to_thread_sync,
    )

    result = await service.analyze_contract("  contract text  ")

    assert result == "analysis: contract text"


@pytest.mark.asyncio
async def test_analyze_contract_wraps_rag_errors(monkeypatch):
    """Verifies that document analysis exceptions are wrapped."""

    service = RAGApplicationService()
    service._status = RAGStatus.READY

    class BrokenAnalysisRAG:
        """Fake RAG engine that fails on contract analysis."""

        def analyze_contract(self, contract_text: str) -> str:
            """Raises a deterministic analysis error."""
            raise ValueError("analysis failed")

    service._rag = BrokenAnalysisRAG()

    monkeypatch.setattr(
        "backend.modules.rag.service.asyncio.to_thread",
        run_to_thread_sync,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await service.analyze_contract("contract")

    assert "Ошибка при анализе документа: analysis failed" in str(exc_info.value)


def test_ensure_ready_raises_when_loading():
    """Verifies that a loading RAG service rejects requests."""

    service = RAGApplicationService()
    service._status = RAGStatus.LOADING

    with pytest.raises(RuntimeError) as exc_info:
        service._ensure_ready()

    assert str(exc_info.value) == "RAG ещё загружается. Попробуйте отправить вопрос чуть позже."


def test_ensure_ready_raises_when_error():
    """Verifies that an errored RAG service rejects requests with details."""

    service = RAGApplicationService()
    service._status = RAGStatus.ERROR
    service._error = "load failed"

    with pytest.raises(RuntimeError) as exc_info:
        service._ensure_ready()

    assert str(exc_info.value) == "RAG недоступен: load failed"


def test_ensure_ready_raises_when_not_initialized():
    """Verifies that an uninitialized RAG service rejects requests."""

    service = RAGApplicationService()

    with pytest.raises(RuntimeError) as exc_info:
        service._ensure_ready()

    assert str(exc_info.value) == "RAG ещё не инициализирован."


def test_health_returns_status_ready_and_error():
    """Verifies that health exposes service status, readiness and error."""

    service = RAGApplicationService()
    service._status = RAGStatus.ERROR
    service._error = "broken"

    assert service.health() == {
        "status": "error",
        "ready": False,
        "error": "broken",
    }
