# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements-vps.txt ./
COPY alphavedha/ alphavedha/

# INSTALL_TARGET=vps installs only serving deps (no torch/transformers/vectorbt).
# INSTALL_TARGET=full installs everything from pyproject.toml (dev/training use).
ARG INSTALL_TARGET=vps
RUN if [ "$INSTALL_TARGET" = "vps" ]; then \
        pip install --no-cache-dir --prefix=/install \
            torch --index-url https://download.pytorch.org/whl/cpu && \
        pip install --no-cache-dir --prefix=/install -r requirements-vps.txt && \
        pip install --no-cache-dir --prefix=/install --no-deps .; \
    else \
        pip install --no-cache-dir --prefix=/install \
            torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
        pip install --no-cache-dir --prefix=/install .; \
    fi

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 curl git openssh-client && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -r alphavedha && \
    useradd -r -g alphavedha -d /app alphavedha

COPY --from=builder /install /usr/local
COPY alphavedha/ alphavedha/
COPY scripts/ scripts/
COPY configs/ configs/
COPY alembic/ alembic/
COPY alembic.ini .

RUN mkdir -p /app/models/artifacts /app/logs && \
    chown -R alphavedha:alphavedha /app

USER alphavedha

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "alphavedha.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
