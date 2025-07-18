# AI Safety Newsletter Agent Dockerfile
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN groupadd --gid 1000 aisafety && \
    useradd --uid 1000 --gid aisafety --shell /bin/bash --create-home aisafety

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.8.3

# Set work directory
WORKDIR /app

# Copy Poetry files
COPY pyproject.toml poetry.lock* ./

# Configure Poetry
RUN poetry config virtualenvs.create false

# Install Python dependencies
RUN poetry install --only=main --no-dev

# Copy application code
COPY aisafety_news/ ./aisafety_news/
COPY models.yaml ./
COPY templates/ ./templates/

# Create necessary directories
RUN mkdir -p data cache/http logs && \
    chown -R aisafety:aisafety /app

# Switch to non-root user
USER aisafety

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import aisafety_news; print('OK')" || exit 1

# Default command
ENTRYPOINT ["python", "-m", "aisafety_news.orchestrator"]
CMD ["--help"]
