# MIDAS GPU Training Project

GPU-accelerated ML training pipeline for high-frequency trading on cryptocurrency markets.

## Quick Start

### 1. Environment Check

First, verify your environment is properly configured:

```bash
python run_env_check.py
```

### 2. Upload Your Data

Upload your feature parquet files to `/data/features/`:

```bash
# Example using rsync
rsync -avz data/features/*.parquet user@vast-instance:/data/features/
```

### 3. Run Training

Start with the linear baseline (fastest):

```bash
python train_linear.py --model-type ridge
```

Then try XGBoost (GPU-accelerated):

```bash
python train_xgboost.py --hyperparameter-search
```

## Project Structure

```
gpu_project/
├── utils.py              # Logging, metrics, plotting utilities
├── dataset.py            # Data loading and feature management
├── preprocessing.py      # Normalization, null handling
├── splitter.py           # OFI-safe data splitting
├── train_xgboost.py      # XGBoost GPU training
├── train_lstm.py         # PyTorch LSTM training
├── train_tft.py          # Temporal Fusion Transformer
├── train_linear.py       # Ridge/Lasso baselines
├── backtest.py           # Walk-forward backtesting
├── run_env_check.py      # Environment verification
└── README.md             # This file
```

## Training Scripts

### Linear Models (Baseline)

```bash
# Ridge regression
python train_linear.py --model-type ridge --hyperparameter-search

# Lasso regression
python train_linear.py --model-type lasso --hyperparameter-search

# Compare all linear models
python train_linear.py --model-type all
```

### XGBoost (GPU)

```bash
# Basic training
python train_xgboost.py

# With hyperparameter search (Optuna)
python train_xgboost.py --hyperparameter-search --n-trials 50

# Custom configuration
python train_xgboost.py \
    --target-type return \
    --target-horizon 10 \
    --train-ratio 0.8
```

### LSTM (GPU)

```bash
# Basic training
python train_lstm.py

# With attention mechanism
python train_lstm.py --use-attention --hidden-size 256

# Custom sequence length
python train_lstm.py --sequence-length 100 --n-epochs 200
```

### TFT (GPU)

```bash
# Basic training
python train_tft.py

# Custom configuration
python train_tft.py --hidden-dim 128 --n-epochs 150
```

## Backtesting

Run walk-forward backtests to evaluate trading performance:

```bash
# XGBoost backtest
python backtest.py --model-type xgboost --n-splits 5

# Ridge backtest
python backtest.py --model-type ridge --n-splits 5

# With custom strategy threshold
python backtest.py --threshold 0.3 --transaction-cost 0.05
```

## Target Types

- `return`: Future return in basis points (regression)
- `direction`: Binary classification (up/down)
- `price_delta`: Raw price difference

```bash
# Direction prediction
python train_xgboost.py --target-type direction

# Different prediction horizons
python train_xgboost.py --target-horizon 5   # 5 ticks = 500ms
python train_xgboost.py --target-horizon 20  # 20 ticks = 2s
```

## Output Structure

Each training run creates a timestamped output directory:

```
outputs/
└── xgboost_20241220_143052/
    ├── model.json            # Trained model
    ├── results.json          # Metrics and configuration
    ├── feature_importance.json
    ├── predictions.png       # Prediction vs actual plot
    ├── feature_importance.png
    └── preprocessor/
        ├── normalization_stats.json
        └── preprocessor_config.json
```

## Key Metrics

### Regression Metrics
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination

### Trading Metrics
- **Directional Accuracy**: % of correct direction predictions
- **Sharpe Ratio**: Risk-adjusted return
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: % of profitable trades

## OFI-Safe Data Splitting

This project uses OFI-safe (Order Flow Imbalance) data splitting to prevent look-ahead bias:

1. **Gap Detection**: Identifies gaps in timestamps (>2x bucket size)
2. **Segment Splitting**: Creates independent segments at gaps
3. **Per-Segment Train/Test**: Each segment split independently
4. **No Cross-Contamination**: OFI cumulative features don't leak

The default bucket size is 100ms, with gap threshold of 200ms.

## GPU Configuration

### CUDA Check
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

### XGBoost GPU
When GPU is available, XGBoost automatically uses:
- `tree_method="gpu_hist"`
- `device="cuda"`

### PyTorch GPU
Models automatically move to CUDA when available.

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch-size 128`
- Reduce hidden size: `--hidden-size 64`
- Reduce sequence length: `--sequence-length 30`

### Slow Training
- Ensure GPU is being used (check env_check.py output)
- Reduce dataset size for testing
- Use smaller hyperparameter search: `--n-trials 10`

### Data Not Found
- Specify data directory: `--data-dir /path/to/features`
- Ensure parquet files exist in the directory

### Package Import Errors
- Install missing packages: `pip install -r requirements.txt`
- Check Python version: `python --version` (needs 3.9+)

## Requirements

```
numpy>=1.20.0
polars>=0.18.0
torch>=2.0.0
xgboost>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
optuna>=3.0.0  # Optional, for hyperparameter search
```

## Example Workflow

1. **Validate environment**
   ```bash
   python run_env_check.py
   ```

2. **Train baseline model**
   ```bash
   python train_linear.py --model-type ridge
   ```

3. **Train XGBoost with tuning**
   ```bash
   python train_xgboost.py --hyperparameter-search
   ```

4. **Backtest best model**
   ```bash
   python backtest.py --model-type xgboost --n-splits 5
   ```

5. **Compare results**
   ```bash
   # Results are in outputs/<model>_<timestamp>/results.json
   ```

## Performance Tips

1. **Start with Ridge**: Fast baseline, good for sanity checks
2. **Use XGBoost for features**: Good feature importance insights
3. **LSTM for sequences**: Captures temporal patterns
4. **Hyperparameter search**: Always use for final models
5. **Walk-forward backtest**: Most realistic performance estimate
