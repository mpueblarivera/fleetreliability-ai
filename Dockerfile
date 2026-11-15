# ==============================================================================
# FleetReliability-AI — Production Dockerfile (Powered by Astral uv)
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    PORT=9696 \
    MODEL_PATH=models/best_model.bin

WORKDIR /app

# Install system-level OpenMP runtime required by LightGBM and XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary directly from official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create a non-root user for security best practices
RUN groupadd -g 10001 -r appgroup && \
    useradd -u 10001 -r -g appgroup -d /app -s /sbin/nologin appuser

# Install dependencies using uv into the container environment
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source code and serialized model artifact
COPY src/ ./src/
COPY models/ ./models/

# Ensure proper permissions for non-root execution
RUN chown -R appuser:appgroup /app
USER appuser

# Expose port 9696 as standard in ML Zoomcamp
EXPOSE 9696

# Healthcheck against the FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; port = os.getenv('PORT', '9696'); urllib.request.urlopen(f'http://localhost:{port}/health')" || exit 1

# Run FastAPI application via pre-synced virtualenv uvicorn with dynamic $PORT
CMD ["sh", "-c", "exec uvicorn src.predict:app --host 0.0.0.0 --port ${PORT:-9696} --workers 2 --access-log"]
