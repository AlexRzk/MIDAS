FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY processor/ ./processor/

# Create data directories
RUN mkdir -p /data/raw /data/clean

# Set environment defaults
ENV PYTHONUNBUFFERED=1
ENV SYMBOL=btcusdt
ENV RAW_DATA_PATH=/data/raw
ENV CLEAN_DATA_PATH=/data/clean
ENV SNAPSHOT_INTERVAL_MS=100
ENV ORDER_BOOK_DEPTH=10
ENV PROCESS_BATCH_SIZE=100000

# Run as non-root user
RUN useradd -r -s /bin/false midas && chown -R midas:midas /app /data
USER midas

CMD ["python", "-m", "processor.main"]
