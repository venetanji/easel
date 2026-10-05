FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Build the package with its declared readme and license materials available.
COPY pyproject.toml uv.lock ./
COPY README.md LICENSE NOTICE ./
COPY licenses ./licenses
COPY easel ./easel
RUN uv sync --frozen --no-dev

EXPOSE 8080
CMD ["uvicorn", "easel.app:app", "--host", "0.0.0.0", "--port", "8080"]
