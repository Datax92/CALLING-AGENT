# Multi-stage Dockerfile for consolidated voice agent, MongoDB, and dashboard

# Stage 1: Build the voice agent and dashboard components
FROM python:3.12-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY requirements-voice.txt requirements-dashboard.txt ./
RUN pip install --no-cache-dir -r requirements-voice.txt -r requirements-dashboard.txt

# Copy application code
COPY . .

# Stage 2: Create the final image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Remove database initialization – MongoDB runs as a managed service
# (no need to create local data directories)


# Copy from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app /app

# Set working directory
WORKDIR /app

# Expose ports
EXPOSE 8000 27017

# Set environment variables
ENV PYTHONPATH=/app \
    MONGO_URI=mongodb://localhost:27017/voicebot \
    DASHBOARD_WEBHOOK_URL=http://localhost:8000/webhook/call-summary

# Start the application
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]