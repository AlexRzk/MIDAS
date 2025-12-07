# MIDAS - Market Intelligence Data Acquisition System

A production-grade crypto data pipeline for collecting, processing, and feature engineering from Binance order book and trade streams. Built for high-frequency trading research and ML model development.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Collector    │────▶│    Processor    │────▶│    Features     │
│     (Rust)      │     │    (Python)     │     │    (Python)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   Raw JSONL.zst         Cleaned Data           Parquet Files
   (data/raw/)           (data/clean/)          (data/features/)
```

### Components

1. **Collector (Rust)**: High-performance WebSocket client collecting L2 depth (100 levels) and trades
   - Automatic reconnection with configurable retry logic
   - ZSTD-compressed JSONL output
   - File rotation by time (hourly) or size (1GB)
   - 24/7 operation with graceful shutdown

2. **Processor (Python/Polars)**: Order book reconstruction from incremental updates
   - Sequence validation and gap detection
   - Full L2 book reconstruction
   - Fixed-interval snapshots (100ms default)
   - Data cleaning and validation

3. **Feature Generator (Python/Polars)**: ML-ready feature computation
   - OFI (Order Flow Imbalance)
   - Book imbalance at multiple depths
   - Microprice
   - Liquidity metrics
   - Trade flow analysis
   - ZSTD-compressed Parquet output

## Quick Start

### Prerequisites
- Docker and Docker Compose
- ~10GB disk space (for data storage)
### Preflight (optional but recommended)

Before starting, run the preflight script to check disk layout and Docker availability:

```bash
chmod +x scripts/preflight.sh && ./scripts/preflight.sh
```
If any directories are missing, the script will print guidance and you can run the setup script:

```bash
chmod +x scripts/setup.sh && ./scripts/setup.sh
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MIDAS.git
cd MIDAS

# Run setup script (creates directories and .env)
# On Windows:
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1

# On Linux/Mac:
chmod +x scripts/setup.sh && ./scripts/setup.sh

# Or manually:
mkdir -p data/raw data/clean data/features
mkdir -p logs/collector logs/processor logs/features
cp .env.example .env
```

### Start the Pipeline

```bash
# Build and start all services
docker compose up -d

# Or using Make:
make up
```

### Monitor the Pipeline

```bash
# View all logs
docker compose logs -f

# View specific service logs
docker compose logs -f collector
docker compose logs -f processor
docker compose logs -f features

# Check service status
docker compose ps
```

### Stop the Pipeline

```bash
docker compose down
```

## Data Flow

### 1. Raw Data (`data/raw/`)
ZSTD-compressed JSONL files containing raw WebSocket messages:

```json
{
  "exchange_ts": 1701936000000000,
  "local_ts": 1701936000001234,
  "type": "depth",
  "symbol": "BTCUSDT",
  "first_update_id": 123456789,
  "last_update_id": 123456790,
  "prev_update_id": 123456788,
  "bids": [["43250.50", "2.5"], ["43250.00", "5.2"]],
  "asks": [["43251.00", "1.8"], ["43251.50", "4.3"]]
}
```

### 2. Clean Data (`data/clean/`)
Intermediate Parquet files with reconstructed order book snapshots at fixed intervals.

### 3. Features (`data/features/`)
ML-ready Parquet files with computed features, ready for training.

## Output Schema

### Core Columns

| Column | Type | Description |
|--------|------|-------------|
| ts | int64 | Timestamp in microseconds (exchange time) |
| bid_px_01 - bid_px_10 | float64 | Best 10 bid prices |
| bid_sz_01 - bid_sz_10 | float64 | Best 10 bid sizes |
| ask_px_01 - ask_px_10 | float64 | Best 10 ask prices |
| ask_sz_01 - ask_sz_10 | float64 | Best 10 ask sizes |

### Derived Features

| Column | Type | Description |
|--------|------|-------------|
| midprice | float64 | (best_bid + best_ask) / 2 |
| spread | float64 | best_ask - best_bid |
| spread_bps | float64 | Spread in basis points |
| microprice | float64 | Size-weighted mid price |

### Order Flow Features

| Column | Type | Description |
|--------|------|-------------|
| imbalance | float64 | (bid_vol - ask_vol) / total_vol at level 1 |
| imbalance_5 | float64 | Imbalance at 5 levels |
| imbalance_10 | float64 | Imbalance at 10 levels |
| ofi | float64 | Order Flow Imbalance: ΔBidSize - ΔAskSize |
| ofi_10 | float64 | Rolling OFI (10 periods) |
| ofi_cumulative | float64 | Cumulative OFI |

### Trade Features

| Column | Type | Description |
|--------|------|-------------|
| taker_buy_volume | float64 | Taker buy volume in bucket |
| taker_sell_volume | float64 | Taker sell volume in bucket |
| signed_volume | float64 | taker_buy - taker_sell |
| volume_imbalance | float64 | Volume imbalance ratio |
| last_trade_px | float64 | Last trade price |
| last_trade_qty | float64 | Last trade quantity |

### Liquidity Metrics

| Column | Type | Description |
|--------|------|-------------|
| liquidity_1 | float64 | Total volume at best level |
| liquidity_5 | float64 | Total volume at 5 levels |
| liquidity_10 | float64 | Total volume at 10 levels |

## OFI Definition

Order Flow Imbalance (OFI) measures the net order flow pressure:

```
OFI_t = (BidSize_t - BidSize_{t-1}) - (AskSize_t - AskSize_{t-1})
```

- **Positive OFI**: Net buying pressure (bid size increasing, ask size decreasing)
- **Negative OFI**: Net selling pressure (ask size increasing, bid size decreasing)

## Configuration

All settings are configured via environment variables in `.env`:

### Symbol Settings
| Variable | Default | Description |
|----------|---------|-------------|
| SYMBOL | btcusdt | Trading pair (lowercase) |

### Collector Settings
| Variable | Default | Description |
|----------|---------|-------------|
| FILE_ROTATION_HOURS | 1 | Rotate files every N hours |
| FILE_ROTATION_GB | 1 | Rotate files at N GB |
| RECONNECT_DELAY_SECS | 5 | Delay between reconnection attempts |
| MAX_RECONNECT_ATTEMPTS | 0 | Max reconnects (0 = infinite) |

### Processor Settings
| Variable | Default | Description |
|----------|---------|-------------|
| SNAPSHOT_INTERVAL_MS | 100 | Order book snapshot interval |
| ORDER_BOOK_DEPTH | 10 | Number of price levels to store |
| PROCESS_BATCH_SIZE | 100000 | Events per processing batch |

### Feature Settings
| Variable | Default | Description |
|----------|---------|-------------|
| TIME_BUCKET_MS | 100 | Feature aggregation bucket |
| OFI_WINDOW | 10 | Rolling OFI window size |
| PARQUET_ROW_GROUP_SIZE | 50000 | Parquet row group size |

## Example Output Row

```json
{
  "ts": 1701936000000000,
  "bid_px_01": 43250.50,
  "bid_sz_01": 2.5,
  "bid_px_02": 43250.00,
  "bid_sz_02": 5.2,
  "ask_px_01": 43251.00,
  "ask_sz_01": 1.8,
  "ask_px_02": 43251.50,
  "ask_sz_02": 4.3,
  "midprice": 43250.75,
  "spread": 0.50,
  "spread_bps": 1.16,
  "imbalance": 0.163,
  "ofi": 0.7,
  "microprice": 43250.66,
  "taker_buy_volume": 15.3,
  "taker_sell_volume": 12.1,
  "signed_volume": 3.2,
  "last_trade_px": 43250.50,
  "last_trade_qty": 0.5,
  "liquidity_1": 4.3,
  "liquidity_5": 25.6,
  "liquidity_10": 48.2
}
```

## Project Structure

```
MIDAS/
├── collector/                 # Rust WebSocket collector
│   ├── Cargo.toml
│   ├── Dockerfile
│   └── src/
│       └── main.rs
├── processor/                 # Python order book processor
│   ├── Dockerfile
│   ├── requirements.txt
│   └── processor/
│       ├── __init__.py
│       ├── config.py
│       ├── reader.py
│       ├── orderbook.py
│       ├── cleaner.py
│       ├── writer.py
│       └── main.py
├── features/                  # Python feature generator
│   ├── Dockerfile
│   ├── requirements.txt
│   └── features/
│       ├── __init__.py
│       ├── config.py
│       ├── compute.py
│       ├── writer.py
│       └── main.py
├── scripts/
│   ├── setup.sh
│   ├── setup.ps1
│   └── inspect_data.py
├── data/                      # Data storage (gitignored)
│   ├── raw/
│   ├── clean/
│   └── features/
├── logs/                      # Service logs
├── docker-compose.yml
├── Makefile
├── .env.example
├── .env
├── .gitignore
└── README.md
```

## Development

### Building Locally

```bash
# Collector (Rust)
cd collector
cargo build --release
./target/release/midas-collector

# Processor (Python)
cd processor
pip install -r requirements.txt
python -m processor.main

# Features (Python)
cd features
pip install -r requirements.txt
python -m features.main
```

### Inspecting Data

```bash
# Using the inspection script
python scripts/inspect_data.py

# Or interactively with Polars
python
>>> import polars as pl
>>> df = pl.read_parquet("data/features/features_*.parquet")
>>> df.head()
```

## Troubleshooting

### Collector not connecting
- Check internet connectivity
- Verify Binance API is accessible from your region
- Check logs: `docker compose logs collector`

### Processor not processing files
- Ensure raw files exist in `data/raw/`
- Files must be closed (not being written to) before processing
- Check logs: `docker compose logs processor`

### High memory usage
- Reduce `PROCESS_BATCH_SIZE`
- Reduce `ORDER_BOOK_DEPTH`

### Missing data gaps
- Check for sequence errors in processor logs
- May indicate network issues during collection

## Performance Notes

- Collector handles ~1000 messages/second easily
- Each hour of BTC data is approximately 50-100MB raw
- Feature files are typically 10-20% of raw size
- Full pipeline latency: <1 second from collection to features

## License

MIT
