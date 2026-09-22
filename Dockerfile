FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /srv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv cache mount
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copy all source modules (app, ingest, data, etc.)
COPY . .

# Final sync to install project code
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Ensure Python imports resolve from /srv root
ENV PYTHONPATH="/srv"

CMD ["uv", "run", "--no-dev", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]