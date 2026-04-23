# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    IS_LOCAL=false

# Install system dependencies for audio processing and ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    ffmpeg \
    libsndfile1 \
    espeak-ng espeak-ng-data libespeak-ng1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
# Install phonemizer FIRST with exact version
RUN pip install --no-cache-dir phonemizer==3.2.1

RUN pip install --no-cache-dir -r requirements.txt

# Create model cache directory
RUN mkdir -p /root/.cache/huggingface/hub

# Copy application code
COPY . .

# Create directories for temporary files
RUN mkdir -p /app/temp /app/logs

EXPOSE 5001

CMD ["python", "-m", "app.main"]