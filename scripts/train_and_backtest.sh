#!/usr/bin/env bash
# Simple training script for MIDAS models
# Usage:
#   ./scripts/train_and_backtest.sh              # Interactive mode
#   ./scripts/train_and_backtest.sh xgboost      # Direct run

set -e

# Defaults
DATA_DIR="data/features_normalized"
OUTPUT_DIR="outputs"
BACKTEST_PER_ITERATION=false
BACKTEST_INTERVAL=10
BACKTEST_METRIC="total_pnl_bps"

# Interactive model selection if no argument provided
if [ $# -eq 0 ]; then
    echo "============================================"
    echo " MIDAS Model Training"
    echo "============================================"
    echo ""
    echo "Select a model to train:"
    echo "  1) XGBoost (GPU-accelerated, fastest, recommended)"
    echo "  2) LSTM (PyTorch, GPU)"
    echo "  3) Linear Ridge (baseline)"
    echo "  4) XGBoost with per-iteration backtest (keep best model via trading metric)"
    echo ""
    read -p "Enter choice [1-3]: " choice
    
    case $choice in
        1) MODEL="xgboost";;
        2) MODEL="lstm";;
        3) MODEL="linear";;
        4) MODEL="xgboost-backtest";;
        *) echo "Invalid choice"; exit 1;;
    esac
else
    MODEL="$1"
fi

echo ""
echo "============================================"
echo " Training: $MODEL"
echo "============================================"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Change to project root
cd "$(dirname "$0")/.."

# Run training based on model type
case "$MODEL" in
    xgboost)
        echo "Starting XGBoost GPU training..."
        python3 training/gpu_project/train_xgboost.py \
            --data-dir "$DATA_DIR" \
            --output-dir "$OUTPUT_DIR/xgboost_$(date +%Y%m%d_%H%M%S)" \
            --target-type return \
            --target-horizon 10 \
            --train-ratio 0.8
        ;;
    
    xgboost-backtest)
        echo "Starting XGBoost with per-iteration backtest..."
        python3 training/gpu_project/train_xgboost.py \
            --data-dir "$DATA_DIR" \
            --output-dir "$OUTPUT_DIR/xgboost_bt_$(date +%Y%m%d_%H%M%S)" \
            --target-type return \
            --target-horizon 10 \
            --train-ratio 0.8 \
            --backtest-per-iteration \
            --backtest-interval $BACKTEST_INTERVAL \
            --backtest-metric $BACKTEST_METRIC
        ;;
    
    xgboost-tuned)
        echo "Starting XGBoost with hyperparameter tuning..."
        python3 training/gpu_project/train_xgboost.py \
            --data-dir "$DATA_DIR" \
            --output-dir "$OUTPUT_DIR/xgboost_tuned_$(date +%Y%m%d_%H%M%S)" \
            --hyperparameter-search \
            --n-trials 50
        ;;
    
    lstm)
        echo "Starting LSTM training..."
        python3 training/gpu_project/train_lstm.py \
            --data-dir "$DATA_DIR" \
            --n-epochs 100 \
            --batch-size 256 \
            --hidden-dim 128
        ;;
    
    linear)
        echo "Starting Linear (Ridge) training..."
        python3 training/gpu_project/train_linear.py \
            --model-type ridge \
            --data-dir "$DATA_DIR" \
            --hyperparameter-search
        ;;
    
    *)
        echo "Unknown model: $MODEL"
        echo "Available models: xgboost, xgboost-tuned, lstm, linear"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo " Training Complete!"
echo "============================================"
echo "Check outputs in: $OUTPUT_DIR/"
echo ""

