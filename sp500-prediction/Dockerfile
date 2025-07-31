# S&P 500 Prediction System - Production Dockerfile
# Multi-stage build for optimized container size

# Stage 1: Build stage
FROM python:3.12-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Stage 2: Production stage
FROM python:3.12-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:/app/src:/app/scheduler" \
    TZ=UTC

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r sp500 && useradd -r -g sp500 -s /bin/bash sp500

# Create application directories
RUN mkdir -p /app/data/raw /app/data/processed /app/models /app/logs /app/config && \
    chown -R sp500:sp500 /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=sp500:sp500 src/ ./src/
COPY --chown=sp500:sp500 scheduler/ ./scheduler/
COPY --chown=sp500:sp500 app/ ./app/
COPY --chown=sp500:sp500 *.py ./
COPY --chown=sp500:sp500 *.md ./

# Copy configuration files
COPY --chown=sp500:sp500 scheduler/scheduler_config.template.json ./config/scheduler_config.template.json

# Create entrypoint script
COPY --chown=sp500:sp500 docker/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Create health check script
COPY --chown=sp500:sp500 docker/healthcheck.py ./healthcheck.py

# Switch to non-root user
USER sp500

# Expose ports
EXPOSE 8501 8502

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python healthcheck.py || exit 1

# Default command
ENTRYPOINT ["./entrypoint.sh"]
CMD ["scheduler"]
