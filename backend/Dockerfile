FROM python:3.12-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1
ENV PIP_NO_CACHE_DIR=1
ENV CMAKE_ARGS="-DLLAMA_NATIVE=OFF"
ENV FORCE_CMAKE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
    && pip install uv

COPY req.backend.txt /app/req.backend.txt
COPY req.rag.txt /app/req.rag.txt
COPY req.test.txt /app/req.test.txt

RUN uv pip install -r /app/req.backend.txt

RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu

RUN uv pip install \
    scikit-build-core \
    cmake \
    ninja

RUN uv pip install llama-cpp-python==0.3.8 --no-build-isolation

RUN uv pip install -r /app/req.rag.txt

RUN uv pip install -r /app/req.test.txt

COPY backend /app/backend
COPY alembic.ini /app/alembic.ini
COPY pytest.ini /app/pytest.ini
COPY migrations /app/migrations

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
