# syntax=docker/dockerfile:1.7

# ---------------------------------------------------------------------------
# Stage 1: builder -- compiles the venv, never shipped in the final image
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: production -- minimal runtime, non-root, read-only friendly
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS production

ARG APP_VERSION=1.0.0
LABEL org.opencontainers.image.title="insar-pinn-ars" \
      org.opencontainers.image.description="InSAR-PINN Artificial Recharge Structure monitoring service" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    ARS_ENVIRONMENT=production

# Minimal runtime deps only -- no compiler toolchain in the final image.
RUN apt-get update && apt-get install --no-install-recommends -y \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 arsapp \
    && useradd --uid 10001 --gid arsapp --shell /usr/sbin/nologin --no-create-home arsapp

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=arsapp:arsapp src/ ./src/
COPY --chown=arsapp:arsapp pyproject.toml ./

RUN mkdir -p /app/artifacts /app/checkpoints \
    && chown -R arsapp:arsapp /app/artifacts /app/checkpoints

USER arsapp
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

ENTRYPOINT ["tini", "--"]
CMD ["uvicorn", "src.backend.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
