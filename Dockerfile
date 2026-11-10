# ==============================================================================
# FleetReliability-AI — Production Dockerfile (Powered by Astral uv)
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install system-level OpenMP runtime required by LightGBM and XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary directly from official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create a non-root user for security best practices
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

# Install dependencies using uv into the container environment
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source code
COPY src/ ./src/

# Ensure proper permissions for non-root execution
RUN chown -R appuser:appgroup /app
USER appuser

# Expose port 9696 as standard in ML Zoomcamp
EXPOSE 9696

# Healthcheck against the FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9696/health')" || exit 1

# Run FastAPI application via uv run
CMD ["uv", "run", "uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "9696"]
