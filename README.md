cd # MIDAS - Market Intelligence Data Acquisition System

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

# Create required directories
mkdir -p data/raw data/clean data/features models reports logs/{collector,processor,features}

# Build and start services
docker compose build
docker compose up -d
```

**Monitor collection:**
```bash
docker compose logs -f collector   # WebSocket collection logs
docker compose logs -f processor   # Order book reconstruction
docker compose logs -f features    # Feature computation
docker compose ps                  # Service status
```

**Check collected data:**
```bash
ls -lh data/features/              # Feature files
du -sh data/features/              # Total size
```

**Stop collection:**
```bash
docker compose down
```

Data is stored in:
- `data/raw/` - Raw order book updates (compressed JSONL)
- `data/clean/` - Reconstructed order book snapshots (Parquet)
- `data/features/` - ML-ready features (Parquet with OHLCV, OFI, imbalance, etc.)

---

## 2. Training on Vast.ai

### Already Collecting Data?

Great! You have feature files like:
```
features_20251208_172242_0000.parquet
features_20251208_172257_0000.parquet
...
```

**Quick validation:**
```bash
# Check how many feature files you have
ls data/features/features_*.parquet | wc -l

# Check total size
du -sh data/features/

# Validate features (optional)
docker compose build tools
docker compose run --rm tools bash -c "pip install -r features/requirements.txt && python scripts/validate_features.py --dir /app/data/features"
```

**Recommendation:** Collect at least 6-12 hours of continuous data before training for better model performance.

### Why Vast.ai?
- GPU instances starting at $0.20/hour
- RTX 4090 / A100 availability
- No long-term commitments

### Step-by-Step Training

#### A. Prepare Data Locally

If you've been collecting data, check your progress:

```bash
# See what you have
ls -lh data/features/

# Count files (each file ≈ 1-2 hours of data)
ls data/features/features_*.parquet | wc -l

# Total data size
du -sh data/features/
```

**Minimum recommended:** 6-12 hours (6-12 files)
**Optimal:** 24+ hours (24+ files)

If you need more data, keep collection running:

```bash
# Check if still collecting
docker compose ps

# If stopped, restart
docker compose up -d

# Monitor
docker compose logs -f features
```

Once you have enough data, proceed to upload.

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
# Back on your local machine, build training image
docker compose -f docker-compose.training.yml build training

# Run backtest
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

**All commands use Docker Compose directly - no Makefile needed!**

| Task | Command |
|------|---------|
| **Setup** | `cp .env.example .env && mkdir -p data/{raw,clean,features} models reports` |
| **Build images** | `docker compose build` |
| **Start collection** | `docker compose up -d` |
| **Stop collection** | `docker compose down` |
| **View logs (all)** | `docker compose logs -f` |
| **View logs (specific)** | `docker compose logs -f <service>` (collector/processor/features) |
| **Service status** | `docker compose ps` |
| **Restart services** | `docker compose restart` |
| **Data stats** | `ls -lh data/features/ && du -sh data/features/` |
| **Run tests** | `docker compose run --rm tools bash -c "pip install -r features/requirements.txt && python -m pytest tests/ -v"` |
| **Validate features** | `docker compose run --rm tools bash -c "pip install -r features/requirements.txt && python scripts/validate_features.py --dir /app/data/features"` |
| **Train (local GPU)** | `docker compose -f docker-compose.training.yml up training` |
| **Train (interactive)** | `docker compose -f docker-compose.training.yml run --rm training bash` |
| **Backtest** | `docker compose -f docker-compose.training.yml run --rm backtest` |
| **Clean data** | `rm -rf data/raw/* data/clean/* data/features/*` |
| **Clean everything** | `docker compose down -v --rmi local` |
| **Shell into container** | `docker compose exec <service> bash` |

**Optional Makefile shortcuts:**
If you prefer, you can still use `make` commands (they wrap the Docker commands above):
- `make up` → `docker compose up -d`
- `make down` → `docker compose down`
- `make logs` → `docker compose logs -f`
- `make test` → runs tests in tools container
- `make train` → starts training
- `make help` → shows all available targets

---

## Troubleshooting

### Issue: Timestamp mismatch (680k rows instead of ~1.2k for 1-minute buckets)

**Cause:** Timestamps in microseconds but treated as milliseconds.

**Fix:** The pipeline auto-detects timestamp units. If you have old data, delete `data/features/` and reprocess:

```bash
# Stop services
docker compose down

# Backup old data (optional)
mv data/features data/features_backup

# Recreate directory
mkdir -p data/features

# Restart to reprocess
docker compose up -d
docker compose logs -f features
```

### Issue: Missing OHLCV columns

**Fix:** Recent code includes OHLC aggregation. Reprocess features:

```bash
docker compose down
rm -rf data/features/*
docker compose up -d features
docker compose logs -f features
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
