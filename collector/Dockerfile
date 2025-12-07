# Build stage
FROM rust:1.83-bookworm AS builder

WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock* ./

# Create dummy src to cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Build dependencies only (cached)
RUN cargo build --release && rm -rf src

# Copy actual source
COPY src ./src

# Build the actual application
RUN touch src/main.rs && cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install CA certificates for TLS
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/midas-collector /app/midas-collector

# Create data directory
RUN mkdir -p /data/raw

# Set environment defaults
ENV RUST_LOG=info
ENV SYMBOL=btcusdt
ENV RAW_DATA_PATH=/data/raw
ENV FILE_ROTATION_HOURS=1
ENV FILE_ROTATION_GB=1
ENV RECONNECT_DELAY_SECS=5
ENV MAX_RECONNECT_ATTEMPTS=0

# Run as non-root user
RUN useradd -r -s /bin/false midas && chown -R midas:midas /app /data
USER midas

CMD ["./midas-collector"]
