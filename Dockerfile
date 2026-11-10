# ==============================================================================
# FleetReliability-AI — Production Dockerfile
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system-level OpenMP runtime required by LightGBM / XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security best practices
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

# Copy dependency list first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model artifacts
COPY src/ ./src/

# Ensure proper permissions for non-root execution
RUN chown -R appuser:appgroup /app
USER appuser

# Expose port 9696 as standard in ML Zoomcamp
EXPOSE 9696

# Healthcheck against the FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9696/health')" || exit 1

# Run FastAPI application via Uvicorn
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "9696"]
