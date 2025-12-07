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
COPY features/ ./features/

# Create data directories
RUN mkdir -p /data/clean /data/features

# Set environment defaults
ENV PYTHONUNBUFFERED=1
ENV SYMBOL=btcusdt
ENV CLEAN_DATA_PATH=/data/clean
ENV FEATURES_DATA_PATH=/data/features
ENV TIME_BUCKET_MS=100
ENV OFI_WINDOW=10
ENV ORDER_BOOK_DEPTH=10
ENV PARQUET_ROW_GROUP_SIZE=50000

# Run as non-root user
RUN useradd -r -s /bin/false midas && chown -R midas:midas /app /data
USER midas

CMD ["python", "-m", "features.main"]
