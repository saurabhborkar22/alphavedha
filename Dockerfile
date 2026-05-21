# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY alphavedha/ alphavedha/

RUN pip install --no-cache-dir --prefix=/install .

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 curl && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -r alphavedha && \
    useradd -r -g alphavedha -d /app alphavedha

COPY --from=builder /install /usr/local
COPY alphavedha/ alphavedha/
COPY configs/ configs/
COPY alembic/ alembic/
COPY alembic.ini .

RUN mkdir -p /app/models/artifacts /app/logs && \
    chown -R alphavedha:alphavedha /app

USER alphavedha

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "alphavedha.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
