#!/usr/bin/env bash
# train_and_backtest.sh
# Usage:
#   ./scripts/train_and_backtest.sh --model tft --epochs 100 --gpus 1 --batch-size 64

set -euo pipefail

# Defaults
MODEL_TYPE="tft"
EPOCHS=100
BATCH_SIZE=64
GPUS=1
DATA_DIR="data/features"
MODEL_DIR="models"
LOG_DIR="logs/tensorboard"

usage() {
  echo "Usage: $0 [--model tft|xgboost|lstm|linear] [--epochs N] [--batch-size N] [--gpus N] [--data-dir PATH] [--model-dir PATH]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL_TYPE="$2"; shift 2;;
    --epochs)
      EPOCHS="$2"; shift 2;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2;;
    --gpus)
      GPUS="$2"; shift 2;;
    --data-dir)
      DATA_DIR="$2"; shift 2;;
    --model-dir)
      MODEL_DIR="$2"; shift 2;;
    -h|--help)
      usage;;
    *)
      echo "Unknown arg: $1"; usage;;
  esac
done

if [[ "$MODEL_TYPE" == "tft" ]]; then
  docker compose -f docker-compose.training.yml run --rm training \
    python training/train.py --data-dir /app/$DATA_DIR --model-dir /app/$MODEL_DIR --log-dir /app/$LOG_DIR \
      --epochs $EPOCHS --batch-size $BATCH_SIZE --gpus $GPUS
elif [[ "$MODEL_TYPE" == "xgboost" ]]; then
  docker compose -f docker-compose.training.yml run --rm training \
    python training/gpu_project/train_xgboost.py --data-dir /app/$DATA_DIR
elif [[ "$MODEL_TYPE" == "lstm" ]]; then
  docker compose -f docker-compose.training.yml run --rm training \
    python training/gpu_project/train_lstm.py --data-dir /app/$DATA_DIR --n-epochs $EPOCHS --batch-size $BATCH_SIZE
elif [[ "$MODEL_TYPE" == "linear" ]]; then
  docker compose -f docker-compose.training.yml run --rm training \
    python training/gpu_project/train_linear.py --model-type ridge --data-dir /app/$DATA_DIR --hyperparameter-search
else
  echo "Unknown model type: $MODEL_TYPE"; exit 1
fi

# After training, run backtest
docker compose -f docker-compose.training.yml run --rm backtest \
  python training/backtest.py --model /app/$MODEL_DIR/best_model.pt --data-dir /app/$DATA_DIR --normalizer /app/$MODEL_DIR/normalizer.json --output /app/reports/backtest_results.csv || \
  echo "Backtest failed. If model path is different, adjust --model path and retry."

echo "Training and backtest finished. Results saved to reports/"

