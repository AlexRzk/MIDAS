# MIDAS - Market Intelligence Data Acquisition System

High-frequency crypto market data pipeline for ML-driven trading. Collects 100ms order book snapshots from Binance, computes microstructure features, and trains Temporal Fusion Transformer (TFT) models for price prediction.

---

## Quick Start

### 1. Data Collection (Local or Server)

**Prerequisites:**
- Docker & Docker Compose
- 50GB+ free disk space

**Setup:**
```bash
git clone https://github.com/AlexRzk/MIDAS.git
cd MIDAS
cp .env.example .env
make setup
make build
make up
```

**Monitor collection:**
```bash
make logs-collector  # WebSocket collection logs
make logs-processor  # Order book reconstruction
make logs-features   # Feature computation
make stats           # Show collected data size
```

**Stop collection:**
```bash
make down
```

Data is stored in:
- `data/raw/` - Raw order book updates (compressed JSONL)
- `data/clean/` - Reconstructed order book snapshots (Parquet)
- `data/features/` - ML-ready features (Parquet with OHLCV, OFI, imbalance, etc.)

---

## 2. Training on Vast.ai

### Why Vast.ai?
- GPU instances starting at $0.20/hour
- RTX 4090 / A100 availability
- No long-term commitments

### Step-by-Step Training

#### A. Prepare Data Locally

Collect at least 12-24 hours of data:

```bash
# Start collection
make up

# Wait 12-24 hours, then check data
make stats

# Validate features
make validate-features

# Stop collection
make down
```

Expected feature file size: ~100-500MB per 24 hours (BTCUSDT, 1-minute aggregation).

#### B. Create Vast.ai Instance

1. Sign up at [vast.ai](https://vast.ai)
2. Add billing ($10 minimum)
3. Search for instances:
   - **GPU:** RTX 4090 (24GB VRAM) or RTX 3090 (24GB)
   - **RAM:** 32GB+ recommended
   - **Disk:** 100GB+
   - **Image:** `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
   - **Ports:** Open port 22 (SSH)

4. Rent instance → Copy SSH command

#### C. Upload Data to Vast.ai

```bash
# From your local machine, get instance IP from vast.ai dashboard
export VAST_IP=<instance_ip>
export VAST_PORT=<ssh_port>

# Upload feature data
scp -P $VAST_PORT -r data/features root@$VAST_IP:/workspace/data/

# Upload training code
scp -P $VAST_PORT -r training root@$VAST_IP:/workspace/
scp -P $VAST_PORT Dockerfile.training docker-compose.training.yml root@$VAST_IP:/workspace/
```

Or use `rsync` for faster transfers:

```bash
rsync -avz -e "ssh -p $VAST_PORT" data/features/ root@$VAST_IP:/workspace/data/features/
rsync -avz -e "ssh -p $VAST_PORT" training/ root@$VAST_IP:/workspace/training/
```

#### D. SSH into Instance and Train

```bash
ssh -p $VAST_PORT root@$VAST_IP

# Inside instance:
cd /workspace

# Verify GPU
nvidia-smi

# Install dependencies
pip install torch pytorch-lightning pytorch-forecasting polars pyarrow pandas scikit-learn tensorboard

# Start training
python training/train.py \
    --data-dir /workspace/data/features \
    --model-dir /workspace/models \
    --log-dir /workspace/logs \
    --epochs 100 \
    --batch-size 64 \
    --gpus 1 \
    --learning-rate 0.001 \
    --input-length 60 \
    --output-length 10

# Monitor with TensorBoard (in another terminal)
tensorboard --logdir=/workspace/logs --host=0.0.0.0 --port=6006

# Access TensorBoard: http://<VAST_IP>:6006 (ensure port 6006 is open)
```

**Training Time Estimates:**
- RTX 4090: ~2-4 hours for 100 epochs (1M samples)
- RTX 3090: ~3-5 hours for 100 epochs
- Cost: ~$0.50-$1.50 total

#### E. Download Trained Model

```bash
# From local machine
scp -P $VAST_PORT root@$VAST_IP:/workspace/models/best_model.pt models/
scp -P $VAST_PORT root@$VAST_IP:/workspace/models/normalizer.json models/
```

#### F. Run Backtest Locally

```bash
# Back on your local machine
docker compose -f docker-compose.training.yml build training

docker compose -f docker-compose.training.yml run --rm backtest \
    python training/backtest.py \
    --model /app/models/best_model.pt \
    --data-dir /app/data/features \
    --output /app/reports/backtest_results.csv

# View results
cat reports/backtest_results.json
```

---

## Alternative: Docker Training on Vast.ai

If you prefer Docker-based training:

```bash
# On Vast.ai instance:
cd /workspace

# Build training image
docker build -f Dockerfile.training -t midas-training .

# Train with Docker
docker run --gpus all \
    -v /workspace/data:/app/data \
    -v /workspace/models:/app/models \
    -v /workspace/logs:/app/logs \
    midas-training \
    python training/train.py \
    --data-dir /app/data/features \
    --model-dir /app/models \
    --epochs 100 \
    --gpus 1
```

---

## Architecture

### Data Pipeline

```
Binance WebSocket → Collector (Rust)
  ↓ (raw/*.jsonl.zst)
Order Book Processor (Python/Polars)
  ↓ (clean/*.parquet)
Feature Generator (Python/Polars)
  ↓ (features/*.parquet)
```

### Features Computed

**Per 1-minute bucket:**
- OHLCV (open, high, low, close, volume, vwap)
- Order Flow Imbalance (OFI)
- Volume imbalances at 1/5/10 levels
- Spread, microprice, liquidity metrics
- Returns, volatility (20/100 periods)
- Advanced: Kyle's lambda, VPIN, queue imbalances

**Total: 72 features per timestep**

### Model

Temporal Fusion Transformer (TFT):
- Input: 60 minutes historical features
- Output: 10 minutes future price (quantile predictions)
- Architecture: LSTM encoder + Multi-head attention + Gated residual networks
- Loss: Quantile regression (P10, P50, P90)

---

## Configuration

### Data Collection

Edit `.env`:

```bash
SYMBOL=btcusdt                    # Trading pair
SNAPSHOT_INTERVAL_MS=100          # Order book snapshot frequency
TIME_BUCKET_MS=60000              # Feature aggregation (60s = 1 min)
ORDER_BOOK_DEPTH=10               # Levels to collect
```

### Training Hyperparameters

See `training/train.py --help`:

```bash
--input-length 60       # Historical timesteps
--output-length 10      # Prediction horizon
--batch-size 64         # Training batch size
--learning-rate 0.001   # Adam learning rate
--hidden-dim 64         # TFT hidden dimension
--n-heads 4             # Attention heads
--dropout 0.1           # Dropout rate
--patience 10           # Early stopping patience
```

---

## Project Structure

```
MIDAS/
├── collector/          # Rust WebSocket collector
├── processor/          # Python order book processor
├── features/           # Feature computation
├── training/           # TFT training pipeline
│   ├── dataset.py      # Data loading
│   ├── model.py        # TFT model
│   ├── train.py        # Training CLI
│   └── backtest.py     # Walk-forward backtester
├── tests/              # Unit tests
├── scripts/            # Validation script
├── data/               # Data storage
│   ├── raw/            # Raw WebSocket data
│   ├── clean/          # Processed order book
│   └── features/       # ML-ready features
├── models/             # Trained models
├── reports/            # Backtest results
└── Makefile            # Common tasks
```

---

## Commands Reference

| Task | Command |
|------|---------|
| Setup | `make setup` |
| Build images | `make build` |
| Start collection | `make up` |
| Stop collection | `make down` |
| View logs | `make logs` |
| Data stats | `make stats` |
| Run tests | `make test` |
| Validate features | `make validate-features` |
| Train (local GPU) | `make train` |
| Backtest | `make backtest` |
| Clean data | `make clean-data` |
| Help | `make help` |

---

## Troubleshooting

### Issue: Timestamp mismatch (680k rows instead of ~1.2k for 1-minute buckets)

**Cause:** Timestamps in microseconds but treated as milliseconds.

**Fix:** The pipeline auto-detects timestamp units. If you have old data, delete `data/features/` and reprocess:

```bash
make clean-data
make up
```

### Issue: Missing OHLCV columns

**Fix:** Recent code includes OHLC aggregation. Reprocess features:

```bash
rm -rf data/features/*
docker compose restart features
```

### Issue: Vast.ai instance out of disk space

**Solution:** Use instances with 100GB+ disk or upload smaller datasets (e.g., 6-12 hours instead of 24).

### Issue: Training OOM (Out of Memory)

**Solutions:**
- Reduce `--batch-size` (try 32 or 16)
- Reduce `--input-length` (try 30 instead of 60)
- Use GPU with more VRAM (RTX 4090 24GB recommended)

---

## Cost Estimates

### Data Collection
- Server: $5-10/month (Hetzner, Digital Ocean, etc.)
- Bandwidth: Negligible (~1GB/day compressed)
- Storage: ~15GB/month (1-minute features)

### Training
- Vast.ai RTX 4090: $0.30/hour × 3 hours = ~$0.90 per training run
- Experimentation (10 runs): ~$9
- Production training (weekly): ~$4/month

**Total: ~$20-30/month for full pipeline + experimentation**

---

## License

MIT

---

## Contributing

PRs welcome for:
- Additional exchanges (Coinbase, Kraken, etc.)
- New features (funding rate, liquidations, whale tracking)
- Model improvements (Transformers, LSTNet, etc.)
- Performance optimizations

---

*Built with Rust, Python, Polars, PyTorch, and Docker.*
