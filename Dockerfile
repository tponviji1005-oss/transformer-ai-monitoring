# Stage 1: Frontend Build Environment
FROM node:22-alpine AS frontend-builder
WORKDIR /build

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Stage 2: Production Python Runtime Environment
FROM python:3.12-slim AS production

# System dependencies for curl health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend source code and trained ML models
COPY src/ /app/src/
COPY models/ /app/models/

# Copy compiled React frontend assets from Stage 1
COPY --from=frontend-builder /build/dist /app/frontend/dist

# Create persistent storage directory for SQLite database
RUN mkdir -p /app/data

# Expose HTTP port
EXPOSE 8000

# Automated container health check against FastAPI health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Production entrypoint: Single Uvicorn worker prevents duplicate background schedulers and SQLite lock contention
CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

