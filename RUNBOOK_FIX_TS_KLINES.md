# MIDAS Pipeline Fix: Timestamps & Klines Runbook

This runbook provides step-by-step commands for deploying the timestamp fixes and kline generation to your MIDAS server.

**Server:** `192.168.1.104`  
**User:** `olo`  
**Project:** `/home/olo/MIDAS` (adjust path as needed)

---

## Prerequisites

Before starting, ensure:
- SSH access to the server
- Docker and Docker Compose installed
- NVIDIA Container Toolkit installed (for GPU training)

---

## 1. Connect to Server

```bash
ssh olo@192.168.1.104
```
*Enter password when prompted*

---

## 2. Navigate to Project Directory

```bash
cd /home/olo/MIDAS
```

---

## 3. Pull Latest Changes

```bash
git fetch origin
git pull origin main
```

If you have local changes to stash:
```bash
git stash
git pull origin main
git stash pop
```

---

## 4. Verify New Files Exist

```bash
ls -la features/features/ts_utils.py
ls -la features/features/kline.py
ls -la scripts/validate_features.py
ls -la training/
ls -la Dockerfile.training
ls -la docker-compose.training.yml
```

Expected files:
- `features/features/ts_utils.py` - Timestamp unit detection
- `features/features/kline.py` - OHLCV kline computation
- `scripts/validate_features.py` - Feature validation
- `training/dataset.py` - PyTorch data loading
- `training/model.py` - TFT model implementation
- `training/train.py` - Training CLI
- `training/backtest.py` - Walk-forward backtester

---

## 5. Rebuild Feature Processor Container

```bash
docker compose build features
```

Or build just the training image:
```bash
docker compose -f docker-compose.training.yml build training
```

---

## 6. Run Unit Tests

```bash
# Enter the features container
docker compose run --rm features bash

# Inside container, run tests
cd /app
python -m pytest tests/ -v

# Exit container
exit
```

Or run directly:
```bash
docker compose run --rm features python -m pytest tests/ -v
```

Expected output:
```
tests/test_ts_unit_detection.py ........ PASSED
tests/test_kline_from_trades.py ....... PASSED
tests/test_feature_presence.py ........ PASSED
```

---

## 7. Validate Existing Feature Files

```bash
# Create reports directory if needed
mkdir -p reports

# Run validation
docker compose run --rm features python scripts/validate_features.py \
    --dir /app/data/features \
    --output /app/reports/validation_report.json
```

Check for issues:
```bash
cat reports/validation_report.json | python -m json.tool
```

Look for:
- `"valid": true` for each file
- No missing required columns
- Reasonable row counts (~1200 rows for 1-minute buckets over 20 hours)

---

## 8. Reprocess Data (If Needed)

If validation shows issues (e.g., 680k rows instead of ~1.2k):

```bash
# Stop current processing
docker compose stop features processor

# Clear old features (back up first!)
mkdir -p data/features_backup
mv data/features/*.parquet data/features_backup/

# Restart processing with fixed code
docker compose up -d processor features
```

Monitor logs:
```bash
docker compose logs -f features
```

---

## 9. Train TFT Model (GPU)

### Option A: Using Docker Compose

```bash
# Build training image
docker compose -f docker-compose.training.yml build training

# Start training
docker compose -f docker-compose.training.yml up training
```

### Option B: Interactive Training

```bash
docker compose -f docker-compose.training.yml run --rm training \
    python training/train.py \
    --data-dir /app/data/features \
    --epochs 100 \
    --batch-size 64 \
    --gpus 1 \
    --learning-rate 0.001
```

### Monitor with TensorBoard

```bash
# Start TensorBoard
docker compose -f docker-compose.training.yml up -d tensorboard

# Access in browser: http://192.168.1.104:6006
```

---

## 10. Run Backtest

After training completes:

```bash
docker compose -f docker-compose.training.yml run --rm backtest \
    python training/backtest.py \
    --model /app/models/best_model.pt \
    --data-dir /app/data/features \
    --output /app/reports/backtest_results.csv
```

View results:
```bash
cat reports/backtest_results.json
```

---

## 11. Verify GPU Access

```bash
# Check NVIDIA driver
nvidia-smi

# Test PyTorch CUDA
docker compose -f docker-compose.training.yml run --rm training \
    python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

---

## Troubleshooting

### Problem: 680k rows instead of ~1.2k

**Cause:** Timestamp unit mismatch (microseconds treated as milliseconds)

**Fix:** The `ts_utils.py` now auto-detects timestamp units. Reprocess:

```bash
docker compose stop features
docker compose up -d features
docker compose logs -f features
```

### Problem: Missing kline columns

**Cause:** Kline computation not integrated

**Fix:** The `compute.py` now includes OHLC aggregation. Verify with:

```bash
docker compose run --rm features python -c "
import polars as pl
df = pl.read_parquet('/app/data/features/BTCUSDT.parquet')
print(df.columns)
print(df.select(['open', 'high', 'low', 'close', 'volume']).head())
"
```

### Problem: GPU not detected in Docker

**Fix:** Install NVIDIA Container Toolkit:

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

### Problem: Import errors in training

**Fix:** Ensure PYTHONPATH is set:

```bash
export PYTHONPATH=/app
python training/train.py --help
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Connect | `ssh olo@192.168.1.104` |
| Pull code | `git pull origin main` |
| Run tests | `docker compose run --rm features python -m pytest tests/ -v` |
| Validate | `docker compose run --rm features python scripts/validate_features.py --dir /app/data/features` |
| Train | `docker compose -f docker-compose.training.yml up training` |
| Backtest | `docker compose -f docker-compose.training.yml run backtest` |
| TensorBoard | `http://192.168.1.104:6006` |
| View logs | `docker compose logs -f features` |

---

## Files Changed

| File | Change |
|------|--------|
| `features/features/ts_utils.py` | NEW - Timestamp unit detection |
| `features/features/kline.py` | NEW - OHLCV computation |
| `features/features/compute.py` | MODIFIED - Fixed time_bucket_aggregate |
| `scripts/validate_features.py` | NEW - Feature validation |
| `tests/*.py` | NEW - Unit tests |
| `training/*.py` | NEW - TFT training pipeline |
| `Dockerfile.training` | NEW - GPU training image |
| `docker-compose.training.yml` | NEW - Training services |

---

*Last updated: $(date)*
