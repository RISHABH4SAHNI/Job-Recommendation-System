# syntax=docker/dockerfile:1

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for building scientific libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies via package metadata
COPY setup.py ./
COPY src ./src
COPY config ./config

RUN pip install --upgrade pip && \
    pip install -e . && \
    python -c "import nltk; nltk.download('punkt', quiet=True)"

# Copy scripts so they are available in the image
COPY scripts ./scripts

# Ensure runtime directories exist
RUN mkdir -p /app/data/raw /app/data/processed /app/data/models

EXPOSE 8000

ENV API_HOST=0.0.0.0 \
    API_PORT=8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


